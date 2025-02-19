import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_add_hardswish_kernel(const float* x, const float* add_input, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float tmp = x[idx] + add_input[idx];
        float hsig = fminf(fmaxf(tmp + 3.0f, 0.0f), 6.0f) / 6.0f;
        out[idx] = tmp * tmp * hsig;
    }
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input) {
    int N = x.numel();
    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (N + threads -1) / threads;

    fused_add_hardswish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), add_input.data_ptr<float>(), out.data_ptr<float>(), N);

    return out;
}
"""

# Define the C++ function declaration
cpp_source = """
torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input);
"""

# Compile the custom CUDA kernel
fused_add_hardswish = load_inline(
    name='fused_add_hardswish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_add_hardswish_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
)

# Define the optimized model
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_add_hardswish = fused_add_hardswish

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = self.fused_add_hardswish.fused_add_hardswish_cuda(x, add_input)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W).cuda(), torch.randn(batch_size, out_channels, D*stride, H*stride, W*stride).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]