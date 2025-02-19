import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D Average Pooling
avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(const float* input, float* output, int batch_size, int channels, int height, int width, int pooled_height, int pooled_width, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * pooled_height * pooled_width;
    
    if (idx < total_elements) {
        int pw = idx % pooled_width;
        int ph = (idx / pooled_width) % pooled_height;
        int c = (idx / (pooled_width * pooled_height)) % channels;
        int n = idx / (pooled_width * pooled_height * channels);

        int h_start = ph * stride - padding;
        int w_start = pw * stride - padding;
        int h_end = min(h_start + kernel_size, height + padding);
        int w_end = min(w_start + kernel_size, width + padding);
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);
        h_end = min(h_end, height);
        w_end = min(w_end, width);

        float sum = 0.0;
        int pool_size = (h_end - h_start) * (w_end - w_start);

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                sum += input[((n * channels + c) * height + h) * width + w];
            }
        }

        output[idx] = sum / pool_size;
    }
}

torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    int pooled_height = (height + 2 * padding - kernel_size) / stride + 1;
    int pooled_width = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, channels, pooled_height, pooled_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * pooled_height * pooled_width + block_size - 1) / block_size;

    avg_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, pooled_height, pooled_width, kernel_size, stride, padding
    );

    return output;
}
"""

avg_pool2d_cpp_source = "torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);"

# Compile the inline CUDA code for 2D Average Pooling
avg_pool2d = load_inline(
    name='avg_pool2d',
    cpp_sources=avg_pool2d_cpp_source,
    cuda_sources=avg_pool2d_source,
    functions=['avg_pool2d_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool2d = avg_pool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool2d.avg_pool2d_cuda(x, self.kernel_size, self.stride, self.padding)