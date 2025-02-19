import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float cdf = 0.5 * (1.0 + tanhf((0.7978845608 * (x[idx] + 0.044715 * x[idx] * x[idx] * x[idx]))));
        out[idx] = x[idx] * cdf;
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for GELU activation
gelu = load_inline(
    name='gelu',
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=['gelu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized implementation of the GELU activation function using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu = gelu

    def forward(self, x):
        return self.gelu.gelu_cuda(x)