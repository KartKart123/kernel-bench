import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool2d_kernel(const float* input, float* output, int batch_size, int channels, int height, int width, int kernel_size, int stride, int padding, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (channels * height * width);
    int channel_idx = (idx % (channels * height * width)) / (height * width);
    int height_idx = (idx % (height * width)) / width;
    int width_idx = idx % width;

    if (idx < batch_size * channels * height * width) {
        float max_val = -INFINITY;
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                int h = height_idx + k * dilation - padding;
                int w = width_idx + l * dilation - padding;
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    max_val = fmaxf(max_val, input[batch_idx * channels * height * width + channel_idx * height * width + h * width + w]);
                }
            }
        }
        if (height_idx % stride == 0 && width_idx % stride == 0) {
            output[batch_idx * channels * (height / stride) * (width / stride) + channel_idx * (height / stride) * (width / stride) + (height_idx / stride) * (width / stride) + width_idx / stride] = max_val;
        }
    }
}

torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros({batch_size, channels, (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1, (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1}, torch::TensorOptions().device(input.device()));

    const int block_size = 256;
    const int num_blocks = (batch_size * channels * height * width + block_size - 1) / block_size;

    maxpool2d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width, kernel_size, stride, padding, dilation);

    return output;
}
"""

maxpool2d_cpp_source = (
    "torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"
)

maxpool2d = load_inline(
    name="maxpool2d",
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["maxpool2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
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