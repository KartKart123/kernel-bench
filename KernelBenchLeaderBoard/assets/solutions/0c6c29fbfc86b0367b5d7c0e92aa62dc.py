import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise subtraction
elementwise_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_subtract_kernel(const float* input, float* output, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] - value;
    }
}

torch::Tensor elementwise_subtract_cuda(torch::Tensor input, float value) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_subtract_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), value, size);

    return output;
}
"""

elementwise_subtract_cpp_source = "torch::Tensor elementwise_subtract_cuda(torch::Tensor input, float value);"

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

# Define the custom CUDA kernel for average pooling
avgpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avgpool_kernel(const float* input, float* output, int batch_size, int channels, int height, int width, int kernel_size) {
    int out_height = height / kernel_size;
    int out_width = width / kernel_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * out_height * out_width) {
        int b = idx / (channels * out_height * out_width);
        int c = (idx / (out_height * out_width)) % channels;
        int oh = (idx / out_width) % out_height;
        int ow = idx % out_width;

        float sum = 0.0f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh * kernel_size + kh;
                int iw = ow * kernel_size + kw;
                sum += input[b * channels * height * width + c * height * width + ih * width + iw];
            }
        }
        output[idx] = sum / (kernel_size * kernel_size);
    }
}

torch::Tensor avgpool_cuda(torch::Tensor input, int kernel_size) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_height = height / kernel_size;
    auto out_width = width / kernel_size;
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * out_height * out_width + block_size - 1) / block_size;

    avgpool_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, kernel_size);

    return output;
}
"""

avgpool_cpp_source = "torch::Tensor avgpool_cuda(torch::Tensor input, int kernel_size);"

# Compile the inline CUDA code for average pooling
avgpool = load_inline(
    name='avgpool',
    cpp_sources=avgpool_cpp_source,
    cuda_sources=avgpool_source,
    functions=['avgpool_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.elementwise_subtract = elementwise_subtract
        self.avgpool = avgpool

    def forward(self, x):
        x = self.conv(x)
        x = self.elementwise_subtract.elementwise_subtract_cuda(x, self.subtract1_value)
        x = torch.tanh(x)
        x = self.elementwise_subtract.elementwise_subtract_cuda(x, self.subtract2_value)
        x = self.avgpool.avgpool_cuda(x, self.kernel_size_pool)
        return x