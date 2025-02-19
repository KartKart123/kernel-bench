import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel that fuses division and LeakyReLU
leaky_relu_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_div_kernel(const float* x, float* out, float divisor, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] / divisor;
        out[idx] = (val > 0) ? val : negative_slope * val;
    }
}

torch::Tensor leaky_relu_div_cuda(torch::Tensor x, float divisor, float negative_slope) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    leaky_relu_div_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), divisor, negative_slope, size);

    return out;
}
"""

leaky_relu_div_cpp_source = """
torch::Tensor leaky_relu_div_cuda(torch::Tensor x, float divisor, float negative_slope);
"""

# Compile the inline CUDA code
leaky_relu_div = load_inline(
    name='leaky_relu_div',
    cpp_sources=leaky_relu_div_cpp_source,
    cuda_sources=leaky_relu_div_source,
    functions=['leaky_relu_div_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses division and LeakyReLU into a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.negative_slope = 0.01
        self.leaky_relu_div = leaky_relu_div

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu_div.leaky_relu_div_cuda(x, self.divisor, self.negative_slope)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]