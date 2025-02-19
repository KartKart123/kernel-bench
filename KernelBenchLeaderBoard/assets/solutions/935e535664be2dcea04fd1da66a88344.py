import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused hardswish and ReLU activation
fused_activation_code = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_activation_kernel(const float* __restrict__ input, float* __restrict__ output, int num_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elements) {
        float x = input[idx];
        // hardswish(x): x * relu6(x + 3) / 6
        // ReLU after hardswish: max(0, hardswish(x))
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);  // ReLU6(x + 3)
        float hardswish = x * relu6 / 6.0f;
        float fused_activation = fmaxf(hardswish, 0.0f);  // ReLU(hardswish)
        output[idx] = fused_activation;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    fused_activation_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_elements);

    return output;
}
"""

fused_activation_cpp = "torch::Tensor fused_activation_cuda(torch::Tensor input);"

# Compile the inline CUDA code for fused activation
fused_activation = load_inline(
    name='fused_activation',
    cpp_sources=fused_activation_cpp,
    cuda_sources=fused_activation_code,
    functions=['fused_activation_cuda'],
    verbose=False,
    extra_cuda_cflags=['-use_fast_math']
)

import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that fuses hardswish and ReLU activations using a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation.fused_activation_cuda(x)
        x = torch.softmax(x, dim=1)
        x = torch.mean(x, dim=[2, 3, 4])
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]