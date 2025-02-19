import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum over dim=1
sum_dim1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int batch_size,
                                int channels,
                                int depth,
                                int height,
                                int width)
{
    int n = blockIdx.x;
    int dd = blockIdx.y;
    int h = blockIdx.z;
    int w = threadIdx.x;

    if (w < width) {
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            int idx_in = (((n * channels + c) * depth + dd) * height + h) * width + w;
            sum += input[idx_in];
        }
        int idx_out = ((n * depth + dd) * height + h) * width + w;
        output[idx_out] = sum;
    }
}

torch::Tensor sum_dim1_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);

    auto output = torch::zeros({batch_size, depth, height, width}, input.options());

    const dim3 blocks(batch_size, depth, height);
    const dim3 threads(width);

    sum_dim1_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                         output.data_ptr<float>(),
                                         batch_size, channels,
                                         depth, height, width);
    return output.unsqueeze(1); // Add dimension at dim=1
}
"""

sum_dim1_cpp_source = """
torch::Tensor sum_dim1_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for sum over dim=1
sum_dim1_module = load_inline(
    name='sum_dim1_module',
    cpp_sources=sum_dim1_cpp_source,
    cuda_sources=sum_dim1_source,
    functions=['sum_dim1_cuda'],
    verbose=True,
)

# Define the optimized model
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        self.sum_dim1 = sum_dim1_module

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = self.sum_dim1.sum_dim1_cuda(x)
        return x

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]