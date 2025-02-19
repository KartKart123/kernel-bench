import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise multiplication
elementwise_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_mul_kernel(const float* x, float* out, float multiplier, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] * multiplier;
    }
}

torch::Tensor elementwise_mul_cuda(torch::Tensor x, float multiplier) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_mul_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), multiplier, size);

    return out;
}
"""

elementwise_mul_cpp_source = "torch::Tensor elementwise_mul_cuda(torch::Tensor x, float multiplier);"

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

# Define the custom CUDA kernel for global average pooling
global_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool_kernel(const float* x, float* out, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        float sum = 0.0f;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                sum += x[idx * height * width + h * width + w];
            }
        }
        out[idx] = sum / (height * width);
    }
}

torch::Tensor global_avg_pool_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);
    auto size = batch_size * channels;
    auto out = torch::zeros({batch_size, channels, 1, 1}, x.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    global_avg_pool_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, channels, height, width);

    return out;
}
"""

global_avg_pool_cpp_source = "torch::Tensor global_avg_pool_cuda(torch::Tensor x);"

# Compile the inline CUDA code for global average pooling
global_avg_pool = load_inline(
    name='global_avg_pool',
    cpp_sources=global_avg_pool_cpp_source,
    cuda_sources=global_avg_pool_source,
    functions=['global_avg_pool_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier
        self.elementwise_mul = elementwise_mul
        self.global_avg_pool = global_avg_pool

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.elementwise_mul.elementwise_mul_cuda(x, self.multiplier)
        x = self.global_avg_pool.global_avg_pool_cuda(x)
        x = self.global_avg_pool.global_avg_pool_cuda(x)
        x = torch.mean(x)
        return x