import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise subtraction
elementwise_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_subtract_kernel(const float* x, float* out, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] - value;
    }
}

torch::Tensor elementwise_subtract_cuda(torch::Tensor x, float value) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_subtract_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), value, size);

    return out;
}
"""

elementwise_subtract_cpp_source = "torch::Tensor elementwise_subtract_cuda(torch::Tensor x, float value);"

# Compile the inline CUDA code for element-wise subtraction
elementwise_subtract = load_inline(
    name='elementwise_subtract',
    cpp_sources=elementwise_subtract_cpp_source,
    cuda_sources=elementwise_subtract_source,
    functions=['elementwise_subtract_cuda'],
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
        out[idx] = val * fmaxf(0.0f, fminf(6.0f, val + 3.0f)) / 6.0f;
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

# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val * tanhf(logf(1.0f + expf(val)));
    }
}

torch::Tensor mish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

mish_cpp_source = "torch::Tensor mish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Mish activation
mish = load_inline(
    name='mish',
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=['mish_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.elementwise_subtract = elementwise_subtract
        self.hardswish = hardswish
        self.mish = mish

    def forward(self, x):
        x = self.conv(x)
        x = self.elementwise_subtract.elementwise_subtract_cuda(x, self.subtract_value)
        x = self.hardswish.hardswish_cuda(x)
        x = self.pool(x)
        x = self.mish.mish_cuda(x)
        return x