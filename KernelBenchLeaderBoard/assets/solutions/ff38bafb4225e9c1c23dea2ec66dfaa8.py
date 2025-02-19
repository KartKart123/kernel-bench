import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

vector_row_mul_cpp_source = """
torch::Tensor vector_row_mul_cuda(torch::Tensor A, torch::Tensor B);
"""

vector_row_mul_cuda_source = R"""
#include <torch/extension.h>

__global__ void vector_row_mul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        C[row * M + col] = A[row] * B[row * M + col];
    }
}

torch::Tensor vector_row_mul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = B.size(0);
    int M = B.size(1);
    auto C = torch::empty_like(B);

    dim3 threads(16, 16);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    vector_row_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M);

    return C;
}
"""

vector_row_mul = load_inline(
    name='vector_row_mul',
    cpp_sources=vector_row_mul_cpp_source,
    cuda_sources=vector_row_mul_cuda_source,
    functions=['vector_row_mul_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.vector_row_mul = vector_row_mul

    def forward(self, A, B):
        return self.vector_row_mul.vector_row_mul_cuda(A, B)

M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed