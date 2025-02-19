import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void group_norm_kernel(const float* input, float* output, float* mean, float* var, int N, int C, int H, int W, int G, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements_per_group = (C / G) * H * W;
    int num_groups = G;

    if (idx < N * num_groups) {
        int n = idx / num_groups;
        int g = idx % num_groups;

        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int c = 0; c < C / G; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int index = n * C * H * W + (g * (C / G) + c) * H * W + h * W + w;
                    float val = input[index];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }

        float group_mean = sum / num_elements_per_group;
        float group_var = (sum_sq / num_elements_per_group) - (group_mean * group_mean);

        for (int c = 0; c < C / G; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int index = n * C * H * W + (g * (C / G) + c) * H * W + h * W + w;
                    output[index] = (input[index] - group_mean) / sqrtf(group_var + eps);
                }
            }
        }

        mean[idx] = group_mean;
        var[idx] = group_var;
    }
}

std::vector<torch::Tensor> group_norm_cuda(torch::Tensor input, int G, float eps) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({N, G}, input.options());
    auto var = torch::zeros({N, G}, input.options());

    const int block_size = 256;
    const int num_blocks = (N * G + block_size - 1) / block_size;

    group_norm_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), N, C, H, W, G, eps);

    return {output, mean, var};
}
"""

group_norm_cpp_source = "std::vector<torch::Tensor> group_norm_cuda(torch::Tensor input, int G, float eps);"

# Compile the inline CUDA code for Group Normalization
group_norm = load_inline(
    name='group_norm',
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=['group_norm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for LogSumExp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void logsumexp_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = N * H * W;

    if (idx < num_elements) {
        int n = idx / (H * W);
        int hw = idx % (H * W);

        float max_val = -INFINITY;
        for (int c = 0; c < C; ++c) {
            int index = n * C * H * W + c * H * W + hw;
            if (input[index] > max_val) {
                max_val = input[index];
            }
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            int index = n * C * H * W + c * H * W + hw;
            sum_exp += expf(input[index] - max_val);
        }

        output[idx] = max_val + logf(sum_exp);
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor input, int dim, bool keepdim) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::zeros({N, 1, H, W}, input.options());

    const int block_size = 256;
    const int num_blocks = (N * H * W + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);

    return output;
}
"""

logsumexp_cpp_source = "torch::Tensor logsumexp_cuda(torch::Tensor input, int dim, bool keepdim);"

# Compile the inline CUDA code for LogSumExp
logsumexp = load_inline(
    name='logsumexp',
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=['logsumexp_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = group_norm
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        self.logsumexp = logsumexp
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm, _, _ = self.group_norm.group_norm_cuda(x_conv, self.groups, self.eps)
        # Tanh
        x_tanh = self.tanh(x_norm)
        # HardSwish
        x_hard_swish = self.hard_swish(x_tanh)
        # Residual Addition
        x_res = x_conv + x_hard_swish
        # LogSumExp
        x_logsumexp = self.logsumexp.logsumexp_cuda(x_res, 1, True)
        return x_logsumexp