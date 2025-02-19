import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* a, const float* b, float* out, int m, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < m && idy < p) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += a[idx * n + i] * b[i * p + idy];
        }
        out[idx * p + idy] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = a.size(1);
    auto p = b.size(1);
    auto out = torch::zeros({m, p}, torch::TensorOptions().device(a.device()));

    const int block_size = 16;
    const dim3 num_blocks((m + block_size - 1) / block_size, (p + block_size - 1) / block_size);

    matmul_kernel<<<num_blocks, dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, p);

    return out;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for batch normalization
batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_kernel(const float* x, float* out, float* mean, float* var, float* weight, float* bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int batch_idx = idx / n;
        int feature_idx = idx % n;
        float x_val = x[idx];
        float mean_val = mean[feature_idx];
        float var_val = var[feature_idx];
        float weight_val = weight[feature_idx];
        float bias_val = bias[feature_idx];
        out[idx] = (x_val - mean_val) / sqrt(var_val + 1e-5) * weight_val + bias_val;
    }
}

torch::Tensor batch_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor weight, torch::Tensor bias) {
    auto m = x.size(0);
    auto n = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (m * n + block_size - 1) / block_size;

    batch_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), m, n);

    return out;
}
"""

batch_norm_cpp_source = (
    "torch::Tensor batch_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code for batch normalization
batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* x, float* out, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int batch_idx = idx / n;
        int feature_idx = idx % n;
        float x_val = x[idx];
        float max_val = -1e10;
        for (int i = 0; i < n; i++) {
            max_val = max(max_val, x[batch_idx * n + i]);
        }
        float sum_val = 0.0f;
        for (int i = 0; i < n; i++) {
            sum_val += exp(x[batch_idx * n + i] - max_val);
        }
        out[idx] = exp(x_val - max_val) / sum_val;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto m = x.size(0);
    auto n = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (m * n + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), m, n);

    return out;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm_module = nn.BatchNorm1d(clusters)
        self.batch_norm_module.weight.data.fill_(1)
        self.batch_norm_module.bias.data.fill_(0)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

        self.matmul = matmul
        self.batch_norm = batch_norm
        self.softmax = softmax

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        assignment = self.matmul.matmul_cuda(x, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm.batch_norm_cuda(assignment, self.batch_norm_module.running_mean, self.batch_norm_module.running_var, self.batch_norm_module.weight, self.batch_norm_module.bias)

        assignment = self.softmax.softmax_cuda(assignment)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        a_sum = torch.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = torch.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK