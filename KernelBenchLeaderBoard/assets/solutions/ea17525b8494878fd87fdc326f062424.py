import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* x, float* out, float add_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] + add_value;
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor x, float add_value) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), add_value, size);

    return out;
}
"""

elementwise_add_cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor x, float add_value);"

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name='elementwise_add',
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=['elementwise_add_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for element-wise multiplication
elementwise_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_mul_kernel(const float* x, float* out, float mul_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] * mul_value;
    }
}

torch::Tensor elementwise_mul_cuda(torch::Tensor x, float mul_value) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_mul_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), mul_value, size);

    return out;
}
"""

elementwise_mul_cpp_source = "torch::Tensor elementwise_mul_cuda(torch::Tensor x, float mul_value);"

# Compile the inline CUDA code for element-wise multiplication
elementwise_mul = load_inline(
    name='elementwise_mul',
    cpp_sources=elementwise_mul_cpp_source,
    cuda_sources=elementwise_mul_source,
    functions=['elementwise_mul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for element-wise minimum
elementwise_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_min_kernel(const float* x, float* out, float min_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fminf(x[idx], min_value);
    }
}

torch::Tensor elementwise_min_cuda(torch::Tensor x, float min_value) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_min_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), min_value, size);

    return out;
}
"""

elementwise_min_cpp_source = "torch::Tensor elementwise_min_cuda(torch::Tensor x, float min_value);"

# Compile the inline CUDA code for element-wise minimum
elementwise_min = load_inline(
    name='elementwise_min',
    cpp_sources=elementwise_min_cpp_source,
    cuda_sources=elementwise_min_source,
    functions=['elementwise_min_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.elementwise_add = elementwise_add
        self.elementwise_mul = elementwise_mul
        self.elementwise_min = elementwise_min

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.elementwise_add.elementwise_add_cuda(x, self.add_value)
        x = self.elementwise_min.elementwise_min_cuda(x, 0.0)
        x = torch.nn.functional.gelu(x)
        x = self.elementwise_mul.elementwise_mul_cuda(x, self.multiply_value)
        return x