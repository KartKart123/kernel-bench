import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = 0.5 * val * (1.0 + tanh(sqrt(2.0 / M_PI) * (val + 0.044715 * pow(val, 3))));
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

# Define the custom CUDA kernel for global average pooling
global_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool_kernel(const float* x, float* out, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        int b = idx / channels;
        int c = idx % channels;
        float sum = 0.0;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                sum += x[b * channels * height * width + c * height * width + h * width + w];
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
    auto out = torch::zeros({batch_size, channels}, x.options());

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
    """
    Optimized model that performs a convolution, applies custom GELU, and then performs custom global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.gelu = gelu
        self.global_avg_pool = global_avg_pool

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.conv(x)
        x = self.gelu.gelu_cuda(x)
        x = self.global_avg_pool.global_avg_pool_cuda(x)
        return x