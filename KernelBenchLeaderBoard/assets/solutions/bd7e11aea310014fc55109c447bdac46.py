import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float exp_x = expf(x[idx]);
        float softplus = logf(1.0f + exp_x);
        out[idx] = x[idx] * tanhf(softplus);
    }
}

torch::Tensor mish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

mish_cpp_source = "torch::Tensor mish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Mish activation
mish = load_inline(
    name='mish',
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=['mish_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for Hardtanh activation
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* x, float* out, float min_val, float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(min_val, fminf(max_val, x[idx]));
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x, float min_val, float max_val) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), min_val, max_val, size);

    return out;
}
"""

hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor x, float min_val, float max_val);"

# Compile the inline CUDA code for Hardtanh activation
hardtanh = load_inline(
    name='hardtanh',
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=['hardtanh_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, applies custom Mish activation, adds a value, 
    applies custom Hardtanh activation, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.mish = mish
        self.hardtanh = hardtanh

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.mish.mish_cuda(x) # Custom Mish activation
        x = x + self.add_value
        x = self.hardtanh.hardtanh_cuda(x, -1.0, 1.0) # Custom Hardtanh activation
        x = x * self.scale # Scaling
        return x