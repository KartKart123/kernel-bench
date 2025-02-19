import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        out[idx] = 0.5 * val * (1.0 + tanhf(0.7978845608028654 * (val + 0.044715 * val * val * val)));
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
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu = gelu
    
    def forward(self, x):
        return self.gelu.gelu_cuda(x)

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim).cuda()]

def get_init_inputs():
    return []