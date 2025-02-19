import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for global average pooling
global_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool_kernel(const float* input, float* output, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        int b = idx / channels;
        int c = idx % channels;
        float sum = 0.0f;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                sum += input[b * channels * height * width + c * height * width + h * width + w];
            }
        }
        output[idx] = sum / (height * width);
    }
}

torch::Tensor global_avg_pool_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros({batch_size, channels, 1, 1}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels + block_size - 1) / block_size;

    global_avg_pool_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}
"""

global_avg_pool_cpp_source = "torch::Tensor global_avg_pool_cuda(torch::Tensor input);"

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

# Define the custom CUDA kernel for log-sum-exp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void logsumexp_kernel(const float* input, float* output, int batch_size, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_val = -INFINITY;
        for (int c = 0; c < channels; ++c) {
            float val = input[idx * channels + c];
            if (val > max_val) {
                max_val = val;
            }
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum_exp += expf(input[idx * channels + c] - max_val);
        }
        output[idx] = max_val + logf(sum_exp);
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output = torch::zeros({batch_size, 1, 1, 1}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    logsumexp_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels);

    return output;
}
"""

logsumexp_cpp_source = "torch::Tensor logsumexp_cuda(torch::Tensor input);"

# Compile the inline CUDA code for log-sum-exp
logsumexp = load_inline(
    name='logsumexp',
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=['logsumexp_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.global_avg_pool = global_avg_pool
        self.logsumexp = logsumexp

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.global_avg_pool.global_avg_pool_cuda(x).view(x.size(0), x.size(1), 1, 1)
        x = x + self.bias
        x = self.logsumexp.logsumexp_cuda(x.view(x.size(0), -1)).view(x.size(0), 1, 1, 1)
        x = torch.sum(x, dim=(2, 3))
        x = x * 10.0
        return x