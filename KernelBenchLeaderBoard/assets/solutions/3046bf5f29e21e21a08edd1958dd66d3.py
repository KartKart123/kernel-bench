import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Swish and HardSwish activations
cpp_source = """
torch::Tensor swish_activation_cuda(torch::Tensor x);
torch::Tensor hardswish_activation_cuda(torch::Tensor x);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_activation_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val)); // Swish activation
    }
}

torch::Tensor swish_activation_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int size = x.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    swish_activation_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        size
    );

    return y;
}

__global__ void hardswish_activation_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float relu6 = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
        y[idx] = val * relu6 / 6.0f; // HardSwish activation
    }
}

torch::Tensor hardswish_activation_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int size = x.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    hardswish_activation_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        size
    );

    return y;
}
"""

# Compile the inline CUDA code for Swish and HardSwish activations
swish_hardswish = load_inline(
    name='swish_hardswish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['swish_activation_cuda', 'hardswish_activation_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized Model that replaces Swish and HardSwish activations with custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.swish_activation = swish_hardswish.swish_activation_cuda
        self.hardswish_activation = swish_hardswish.hardswish_activation_cuda

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.swish_activation(x)
        x = self.group_norm(x)
        x = self.hardswish_activation(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]