import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise scaling
elementwise_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_scale_kernel(const float* in, float* out, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * scale;
    }
}

torch::Tensor elementwise_scale_cuda(torch::Tensor in, float scale) {
    auto size = in.numel();
    auto out = torch::zeros_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_scale_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), scale, size);

    return out;
}
"""

elementwise_scale_cpp_source = "torch::Tensor elementwise_scale_cuda(torch::Tensor in, float scale);"

# Compile the inline CUDA code for element-wise scaling
elementwise_scale = load_inline(
    name='elementwise_scale',
    cpp_sources=elementwise_scale_cpp_source,
    cuda_sources=elementwise_scale_source,
    functions=['elementwise_scale_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for global average pooling
global_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool_kernel(const float* in, float* out, int batch_size, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        int b = idx / channels;
        int c = idx % channels;
        float sum = 0.0f;
        for (int d = 0; d < depth; ++d) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    sum += in[b * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w];
                }
            }
        }
        out[idx] = sum / (depth * height * width);
    }
}

torch::Tensor global_avg_pool_cuda(torch::Tensor in) {
    auto batch_size = in.size(0);
    auto channels = in.size(1);
    auto depth = in.size(2);
    auto height = in.size(3);
    auto width = in.size(4);
    auto out = torch::zeros({batch_size, channels, 1, 1, 1}, in.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

    global_avg_pool_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), batch_size, channels, depth, height, width);

    return out;
}
"""

global_avg_pool_cpp_source = "torch::Tensor global_avg_pool_cuda(torch::Tensor in);"

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
    Optimized model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.elementwise_scale = elementwise_scale
        self.global_avg_pool = global_avg_pool

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.elementwise_scale.elementwise_scale_cuda(x, self.scale_factor)
        x = self.batch_norm(x)
        x = self.global_avg_pool.global_avg_pool_cuda(x)
        return x