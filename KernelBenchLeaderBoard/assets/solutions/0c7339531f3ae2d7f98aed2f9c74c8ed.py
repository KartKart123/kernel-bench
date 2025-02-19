import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise multiplication
elementwise_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_mul_kernel(const float* x, float scale, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] * scale;
    }
}

torch::Tensor elementwise_mul_cuda(torch::Tensor x, float scale) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_mul_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), scale, out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_mul_cpp_source = "torch::Tensor elementwise_mul_cuda(torch::Tensor x, float scale);"

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

# Define the custom CUDA kernel for clamping
clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_kernel(const float* x, float min_val, float max_val, float* out, int size) {
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

    clamp_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), min_val, max_val, out.data_ptr<float>(), size);

    return out;
}
"""

clamp_cpp_source = "torch::Tensor clamp_cuda(torch::Tensor x, float min_val, float max_val);"

# Compile the inline CUDA code for clamping
clamp = load_inline(
    name='clamp',
    cpp_sources=clamp_cpp_source,
    cuda_sources=clamp_source,
    functions=['clamp_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        self.elementwise_mul = elementwise_mul
        self.clamp = clamp

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.elementwise_mul.elementwise_mul_cuda(x, self.scale)
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        x = self.clamp.clamp_cuda(x, self.clamp_min, self.clamp_max)
        return x