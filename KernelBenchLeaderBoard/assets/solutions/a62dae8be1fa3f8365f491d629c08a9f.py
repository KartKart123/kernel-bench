import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matrix_multiply_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_multiply_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    auto C = torch::zeros({M, N}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));

    const int block_size_x = 16;
    const int block_size_y = 16;
    const int num_blocks_x = (M + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (N + block_size_y - 1) / block_size_y;

    matrix_multiply_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size_x, block_size_y)>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matrix_multiply_cpp_source = "torch::Tensor matrix_multiply_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
matrix_multiply = load_inline(
    name='matrix_multiply',
    cpp_sources=matrix_multiply_cpp_source,
    cuda_sources=matrix_multiply_source,
    functions=['matrix_multiply_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_multiply = matrix_multiply

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_multiply.matrix_multiply_cuda(A, B)