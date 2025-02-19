import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / (1.0f + expf(-x[idx]));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

swish_cpp_source = "torch::Tensor swish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Swish activation
swish = load_inline(
    name='swish',
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=['swish_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for HardSwish activation
hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardswish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val <= -3.0f) {
            out[idx] = 0.0f;
        } else if (val >= 3.0f) {
            out[idx] = val;
        } else {
            out[idx] = val * (val + 3.0f) / 6.0f;
        }
    }
}

torch::Tensor hardswish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardswish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

hardswish_cpp_source = "torch::Tensor hardswish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for HardSwish activation
hardswish = load_inline(
    name='hardswish',
    cpp_sources=hardswish_cpp_source,
    cuda_sources=hardswish_source,
    functions=['hardswish_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.swish = swish
        self.hardswish = hardswish

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.swish.swish_cuda(x)  # Swish activation
        x = self.group_norm(x)
        x = self.hardswish.hardswish_cuda(x)  # HardSwish activation
        return x