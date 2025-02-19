import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* x, float* out, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] > 0 ? x[idx] : x[idx] * negative_slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, negative_slope);

    return out;
}
"""

leaky_relu_cpp_source = "torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope);"

# Compile the inline CUDA code for LeakyReLU
leaky_relu = load_inline(
    name='leaky_relu',
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=['leaky_relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for Clamp
clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_kernel(const float* x, float* out, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fminf(fmaxf(x[idx], min_val), max_val);
    }
}

torch::Tensor clamp_cuda(torch::Tensor x, float min_val, float max_val) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, min_val, max_val);

    return out;
}
"""

clamp_cpp_source = "torch::Tensor clamp_cuda(torch::Tensor x, float min_val, float max_val);"

# Compile the inline CUDA code for Clamp
clamp = load_inline(
    name='clamp',
    cpp_sources=clamp_cpp_source,
    cuda_sources=clamp_source,
    functions=['clamp_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        out[idx] = 0.5 * x_val * (1.0 + tanhf(0.7978845608 * (x_val + 0.044715 * x_val * x_val * x_val)));
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for GELU
gelu = load_inline(
    name='gelu',
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=['gelu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.leaky_relu = leaky_relu
        self.clamp = clamp
        self.gelu = gelu

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu.leaky_relu_cuda(x, 0.2)
        x = x + self.sum_tensor
        x = self.clamp.clamp_cuda(x, -1.0, 1.0)
        x = self.gelu.gelu_cuda(x)
        return x