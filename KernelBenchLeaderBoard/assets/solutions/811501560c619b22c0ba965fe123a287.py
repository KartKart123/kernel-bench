import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA code for fused Mish and Tanh activation
cuda_source = """
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mish_tanh_activation_kernel(const float* __restrict__ input, float* __restrict__ output, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        float x = input[idx];
        float sp = logf(1.0f + expf(x)); // softplus(x) = ln(1 + exp(x))
        float mish = x * tanhf(sp); // mish(x) = x * tanh(softplus(x))
        output[idx] = tanhf(mish); // tanh(mish(x))
    }
}

torch::Tensor mish_tanh_activation_cuda(torch::Tensor input)
{
    const int num_elements = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    mish_tanh_activation_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_elements);

    return output;
}
"""

cpp_source = """
torch::Tensor mish_tanh_activation_cuda(torch::Tensor input);
"""

# Compile the custom CUDA kernel
activation = load_inline(
    name='mish_tanh_activation',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['mish_tanh_activation_cuda'],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution and applies fused Mish and Tanh activations using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.activation = activation.mish_tanh_activation_cuda

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

batch_size = 16
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]