import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardTanh
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val < min_val) {
            output[idx] = min_val;
        } else if (val > max_val) {
            output[idx] = max_val;
        } else {
            output[idx] = val;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, min_val, max_val);

    return output;
}
"""

hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);"

# Compile the inline CUDA code for HardTanh
hardtanh = load_inline(
    name='hardtanh',
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=['hardtanh_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for mean operation
mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        float sum = 0.0f;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                sum += input[idx * height * width + h * width + w];
            }
        }
        output[idx] = sum / (height * width);
    }
}

torch::Tensor mean_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros({batch_size, channels, 1, 1}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

    mean_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}
"""

mean_cpp_source = "torch::Tensor mean_cuda(torch::Tensor input);"

# Compile the inline CUDA code for mean operation
mean = load_inline(
    name='mean',
    cpp_sources=mean_cpp_source,
    cuda_sources=mean_source,
    functions=['mean_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, followed by max pooling, custom HardTanh activation, custom mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = hardtanh
        self.mean = mean

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh.hardtanh_cuda(x, hardtanh_min, hardtanh_max)
        x = self.mean.mean_cuda(x)
        x = torch.tanh(x)
        return x