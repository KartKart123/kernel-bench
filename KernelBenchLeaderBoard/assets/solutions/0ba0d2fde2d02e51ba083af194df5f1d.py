import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise division and LeakyReLU
elementwise_div_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_div_leaky_relu_kernel(const float* x, float* out, float divisor, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = x[idx] / divisor;
        out[idx] = value > 0 ? value : value * negative_slope;
    }
}

torch::Tensor elementwise_div_leaky_relu_cuda(torch::Tensor x, float divisor, float negative_slope) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_div_leaky_relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), divisor, negative_slope, size);

    return out;
}
"""

elementwise_div_leaky_relu_cpp_source = "torch::Tensor elementwise_div_leaky_relu_cuda(torch::Tensor x, float divisor, float negative_slope);"

# Compile the inline CUDA code for element-wise division and LeakyReLU
elementwise_div_leaky_relu = load_inline(
    name='elementwise_div_leaky_relu',
    cpp_sources=elementwise_div_leaky_relu_cpp_source,
    cuda_sources=elementwise_div_leaky_relu_source,
    functions=['elementwise_div_leaky_relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, divides by a constant, and applies LeakyReLU using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.elementwise_div_leaky_relu = elementwise_div_leaky_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.elementwise_div_leaky_relu.elementwise_div_leaky_relu_cuda(x, self.divisor, 0.01)
        return x