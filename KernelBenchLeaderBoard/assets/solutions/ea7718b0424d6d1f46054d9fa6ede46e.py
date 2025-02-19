import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the optimized operations
custom_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_kernel(const float* input, float* output, int size, float add_value, float multiply_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] + add_value;
        val = fminf(val, 0.0f);  // Min with 0
        val = val * 0.5f * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));  // GELU approximation
        output[idx] = val * multiply_value;
    }
}

torch::Tensor custom_cuda(torch::Tensor input, float add_value, float multiply_value) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, add_value, multiply_value);

    return output;
}
"""

custom_kernel_cpp_source = "torch::Tensor custom_cuda(torch::Tensor input, float add_value, float multiply_value);"

# Compile the inline CUDA code for the custom operations
custom_kernel = load_inline(
    name='custom_kernel',
    cpp_sources=custom_kernel_cpp_source,
    cuda_sources=custom_kernel_source,
    functions=['custom_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for post-convolution operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.custom_kernel = custom_kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.custom_kernel.custom_cuda(x, self.add_value, self.multiply_value)
        return x

batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]