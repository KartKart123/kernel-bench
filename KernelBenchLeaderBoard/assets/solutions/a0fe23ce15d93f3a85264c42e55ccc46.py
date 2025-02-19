import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-scalar multiplication
matrix_scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_scalar_mul_kernel(const float* A, float s, float* C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < M && idy < N) {
        C[idx * N + idy] = A[idx * N + idy] * s;
    }
}

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s) {
    auto M = A.size(0);
    auto N = A.size(1);
    auto C = torch::zeros_like(A);

    const int block_size_x = 16;
    const int block_size_y = 16;
    const int num_blocks_x = (M + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (N + block_size_y - 1) / block_size_y;

    matrix_scalar_mul_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size_x, block_size_y)>>>(A.data_ptr<float>(), s, C.data_ptr<float>(), M, N);

    return C;
}
"""

matrix_scalar_mul_cpp_source = "torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s);"

# Compile the inline CUDA code for matrix-scalar multiplication
matrix_scalar_mul = load_inline(
    name='matrix_scalar_mul',
    cpp_sources=matrix_scalar_mul_cpp_source,
    cuda_sources=matrix_scalar_mul_source,
    functions=['matrix_scalar_mul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_scalar_mul = matrix_scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.matrix_scalar_mul.matrix_scalar_mul_cuda(A, s)