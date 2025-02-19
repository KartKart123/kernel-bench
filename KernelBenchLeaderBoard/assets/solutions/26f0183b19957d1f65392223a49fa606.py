import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the optimized operations
custom_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_kernel(
    const float* input, const float* bias, float* output, 
    int batch_size, int out_channels, int height, int width, 
    float scaling_factor) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * out_channels * height * width;
    
    if (idx < total_size) {
        int c = (idx / (height * width)) % out_channels;
        float val = input[idx] + bias[c];
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        val *= scaling_factor;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        output[idx] = val / scaling_factor;
    }
}

torch::Tensor custom_cuda(
    torch::Tensor input, torch::Tensor bias, float scaling_factor) {
    
    auto output = torch::zeros_like(input);
    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    int total_size = batch_size * out_channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    custom_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), 
        batch_size, out_channels, height, width, scaling_factor);
    
    return output;
}
"""

custom_cpp_source = "torch::Tensor custom_cuda(torch::Tensor input, torch::Tensor bias, float scaling_factor);"

# Compile the inline CUDA code
custom_op = load_inline(
    name='custom_op',
    cpp_sources=custom_cpp_source,
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.custom_op = custom_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.custom_op.custom_cuda(x, self.bias, self.scaling_factor)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]