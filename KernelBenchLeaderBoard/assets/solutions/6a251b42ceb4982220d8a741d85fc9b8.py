import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM + GELU
gemm_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gemm_gelu_kernel(const float* a, const float* weight, const float* bias, float* out, int batch_size, int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * out_features) {
        float sum = 0;
        int batch_index = i / out_features;
        int out_index = i % out_features;
        for (int j = 0; j < in_features; ++j) {
            sum += a[batch_index * in_features + j] * weight[out_index * in_features + j];
        }
        sum += bias[out_index];
        out[i] = sum * 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * (sum + 0.044715 * sum * sum * sum)));
    }
}

torch::Tensor gemm_gelu_cuda(torch::Tensor a, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = a.size(0);
    int in_features = a.size(1);
    int out_features = weight.size(0);
    auto out = torch::zeros({batch_size, out_features}, a.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

    gemm_gelu_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features);

    return out;
}
"""

gemm_gelu_cpp_source = "torch::Tensor gemm_gelu_cuda(torch::Tensor a, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code for GEMM + GELU
gemm_gelu = load_inline(
    name='gemm_gelu',
    cpp_sources=gemm_gelu_cpp_source,
    cuda_sources=gemm_gelu_source,
    functions=['gemm_gelu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, GroupNorm, Mean, and ReLU operations in sequence.
    """
    def __init__(self, in_features, out_features, num_groups):
        super(ModelNew, self).__init__()
        self.gemm_gelu = gemm_gelu
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias = nn.Parameter(torch.randn(out_features))
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm_gelu.gemm_gelu_cuda(x, self.gemm_weight, self.gemm_bias)
        x = self.batch_norm(x)
        x = self.group_norm(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.relu(x)
        return x