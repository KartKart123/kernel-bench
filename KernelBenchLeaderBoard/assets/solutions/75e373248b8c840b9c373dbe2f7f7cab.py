import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus activation
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softplus_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = log1p(exp(x[idx]));
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Softplus activation
softplus = load_inline(
    name='softplus',
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=['softplus_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus = softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return self.softplus.softplus_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed