import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D Average Pooling
avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_kernel(const float* input, float* output, int batch_size, int channels, int height, int width, int kernel_size, int stride, int padding) {
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * channels * out_height * out_width) return;

    int out_w = out_idx % out_width;
    int out_h = (out_idx / out_width) % out_height;
    int out_c = (out_idx / (out_width * out_height)) % channels;
    int out_b = out_idx / (out_width * out_height * channels);

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = out_h * stride + kh - padding;
            int in_w = out_w * stride + kw - padding;

            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                int in_idx = (out_b * channels + out_c) * height * width + in_h * width + in_w;
                sum += input[in_idx];
                count += 1;
            }
        }
    }

    output[out_idx] = sum / count;
}

torch::Tensor avg_pool_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * out_height * out_width + block_size - 1) / block_size;

    avg_pool_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, kernel_size, stride, padding);

    return output;
}
"""

avg_pool_cpp_source = "torch::Tensor avg_pool_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"

# Compile the inline CUDA code for 2D Average Pooling
avg_pool = load_inline(
    name='avg_pool',
    cpp_sources=avg_pool_cpp_source,
    cuda_sources=avg_pool_source,
    functions=['avg_pool_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool = avg_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool.avg_pool_cuda(x, self.kernel_size, self.stride, self.padding)