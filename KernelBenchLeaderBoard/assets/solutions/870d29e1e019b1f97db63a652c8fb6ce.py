import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# CUDA code for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* x, float* y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float xi = x[idx];
        float x3 = xi * xi * xi;
        const float c0 = 0.044715f;
        const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
        float tanharg = sqrt_2_over_pi * (xi + c0 * x3);
        float tanhres = tanhf(tanharg);
        y[idx] = 0.5f * xi * (1.0f + tanhres);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x)
{
    auto y = torch::empty_like(x);
    int size = x.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_cuda(torch::Tensor x);
"""

# Compile the CUDA code
gelu = load_inline(
    name='gelu_cuda',
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=['gelu_cuda'],
    verbose=True
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