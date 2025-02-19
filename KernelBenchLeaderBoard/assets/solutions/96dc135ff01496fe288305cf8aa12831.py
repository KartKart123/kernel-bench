import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and function for scalar multiplication
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scalar_multiply_kernel(const float* A, float s, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor scalar_multiply_cuda(torch::Tensor A, double s) {
    auto size = A.numel();
    auto C = torch::zeros_like(A);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    scalar_multiply_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        static_cast<float>(s),
        C.data_ptr<float>(),
        size);

    return C;
}
"""

cpp_source = "torch::Tensor scalar_multiply_cuda(torch::Tensor A, double s);"

# Compile the inline CUDA code for scalar multiplication
scalar_multiply = load_inline(
    name='scalar_multiply',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['scalar_multiply_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scalar_multiply = scalar_multiply

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.scalar_multiply.scalar_multiply_cuda(A, s)

M = 16384
N = 4096

def get_inputs():
    A = torch.randn(M, N).cuda()
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []