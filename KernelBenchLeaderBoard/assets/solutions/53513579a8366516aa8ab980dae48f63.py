import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 2D
maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool2d_kernel(const float* input, float* output, int batch_size, int channels, int height, int width, 
                                 int kernel_size, int stride, int padding, int dilation, int out_height, int out_width) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    if (out_x < out_width && out_y < out_height) {
        float max_val = -FLT_MAX;
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = (out_x * stride - padding) + kx * dilation;
                int in_y = (out_y * stride - padding) + ky * dilation;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    float val = input[(out_c * height + in_y) * width + in_x];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        output[(out_c * out_height + out_y) * out_width + out_x] = max_val;
    }
}

torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());

    const int block_size_x = 16;
    const int block_size_y = 16;
    const int grid_size_x = (out_width + block_size_x - 1) / block_size_x;
    const int grid_size_y = (out_height + block_size_y - 1) / block_size_y;

    for (int b = 0; b < batch_size; ++b) {
        maxpool2d_kernel<<<dim3(grid_size_x, grid_size_y, channels), dim3(block_size_x, block_size_y)>>>(
            input[b].data_ptr<float>(), output[b].data_ptr<float>(), batch_size, channels, height, width, 
            kernel_size, stride, padding, dilation, out_height, out_width);
    }

    return output;
}
"""

maxpool2d_cpp_source = "torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

# Compile the inline CUDA code for Max Pooling 2D
maxpool2d = load_inline(
    name='maxpool2d',
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=['maxpool2d_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d = maxpool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d.maxpool2d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)