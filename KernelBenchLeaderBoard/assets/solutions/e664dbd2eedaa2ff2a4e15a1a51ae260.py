import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for mish activation with subtraction
mish_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_subtract_kernel(const float* __restrict__ input, float* __restrict__ output, float subtract_value, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = input[idx] - subtract_value;
        float sp = logf(1.0f + expf(x)); // softplus
        float mish = x * tanhf(sp);
        output[idx] = mish;
    }
}

torch::Tensor mish_subtract_cuda(torch::Tensor input, float subtract_value)
{
    auto output = torch::empty_like(input);
    int num_elements = input.numel();

    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    mish_subtract_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), subtract_value, num_elements);

    return output;
}
"""

mish_subtract_cpp_source = "torch::Tensor mish_subtract_cuda(torch::Tensor input, float subtract_value);"

# Compile the inline CUDA code
mish_subtract = load_inline(
    name='mish_subtract',
    cpp_sources=mish_subtract_cpp_source,
    cuda_sources=mish_subtract_source,
    functions=['mish_subtract_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, subtracts two values, applies Mish activation using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.total_subtract_value = subtract_value_1 + subtract_value_2
        self.mish_subtract = mish_subtract

    def forward(self, x):
        x = self.conv(x)
        x = self.mish_subtract.mish_subtract_cuda(x, self.total_subtract_value)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]