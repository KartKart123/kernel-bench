import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for clamping and dividing
clamp_and_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_and_divide_kernel(const float* x, float* out, int size, float min_value, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val < min_value) {
            val = min_value;
        }
        out[idx] = val / divisor;
    }
}

torch::Tensor clamp_and_divide_cuda(torch::Tensor x, float min_value, float divisor) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_and_divide_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, min_value, divisor);

    return out;
}
"""

clamp_and_divide_cpp_source = "torch::Tensor clamp_and_divide_cuda(torch::Tensor x, float min_value, float divisor);"

# Compile the inline CUDA code for clamping and dividing
clamp_and_divide = load_inline(
    name='clamp_and_divide',
    cpp_sources=clamp_and_divide_cpp_source,
    cuda_sources=clamp_and_divide_source,
    functions=['clamp_and_divide_cuda'],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel for clamping and dividing.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.clamp_and_divide = clamp_and_divide

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.clamp_and_divide.clamp_and_divide_cuda(x, self.min_value, self.divisor)
        return x