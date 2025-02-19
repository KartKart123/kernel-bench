import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the activation function
activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void activation_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float val_sp = logf(1.0f + expf(val));  // softplus(val)
        float val_tanh_sp = tanhf(val_sp);      // tanh(softplus(val))
        out[idx] = val * val_tanh_sp;           // x * tanh(softplus(x))
    }
}

torch::Tensor activation_cuda(torch::Tensor x) {
    const auto size = x.numel();
    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    activation_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

activation_cpp_source = "torch::Tensor activation_cuda(torch::Tensor x);"

# Compile the inline CUDA code for the activation function
activation = load_inline(
    name='activation',
    cpp_sources=activation_cpp_source,
    cuda_sources=activation_source,
    functions=['activation_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation.activation_cuda(x)
        x = self.bn(x)
        return x