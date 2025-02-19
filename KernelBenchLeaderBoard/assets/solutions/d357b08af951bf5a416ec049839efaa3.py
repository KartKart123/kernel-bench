import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the optimized operations
custom_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_kernel(
    const float* input, float* output, int size, float add_value, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply Mish activation
        float x = input[idx];
        float mish = x * tanhf(log1p(exp(x)));
        
        // Add value
        mish += add_value;
        
        // Apply Hardtanh activation
        mish = fminf(fmaxf(mish, -1.0f), 1.0f);
        
        // Scale the output
        output[idx] = mish * scale;
    }
}

torch::Tensor custom_cuda_op(torch::Tensor input, float add_value, float scale) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    custom_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size, add_value, scale);

    return output;
}
"""

custom_kernel_cpp_source = "torch::Tensor custom_cuda_op(torch::Tensor input, float add_value, float scale);"

# Compile the inline CUDA code for the custom operations
custom_op = load_inline(
    name='custom_op',
    cpp_sources=custom_kernel_cpp_source,
    cuda_sources=custom_kernel_source,
    functions=['custom_cuda_op'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for Mish activation, addition, Hardtanh, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.custom_op = custom_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.custom_op.custom_cuda_op(x, self.add_value, self.scale)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]