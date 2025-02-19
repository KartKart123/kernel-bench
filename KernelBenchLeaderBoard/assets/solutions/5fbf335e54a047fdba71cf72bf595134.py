import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for clamping and division
clamp_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_divide_kernel(const float* input, float* output, float min_value, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val < min_value) {
            val = min_value;
        }
        output[idx] = val / divisor;
    }
}

torch::Tensor clamp_divide_cuda(torch::Tensor input, float min_value, float divisor) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_divide_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), min_value, divisor, size);

    return output;
}
"""

clamp_divide_cpp_source = "torch::Tensor clamp_divide_cuda(torch::Tensor input, float min_value, float divisor);"

# Compile the inline CUDA code for clamping and division
clamp_divide = load_inline(
    name='clamp_divide',
    cpp_sources=clamp_divide_cpp_source,
    cuda_sources=clamp_divide_source,
    functions=['clamp_divide_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.clamp_divide = clamp_divide

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.clamp_divide.clamp_divide_cuda(x, self.min_value, self.divisor)
        return x