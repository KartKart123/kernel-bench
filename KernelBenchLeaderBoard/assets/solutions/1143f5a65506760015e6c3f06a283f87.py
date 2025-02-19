import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_kernel_code = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Block size
    const int TILE_SIZE = 16;

    // Block row and column
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread row and column within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row index of C and A
    int row = by * TILE_SIZE + ty;
    // Column index of C and B
    int col = bx * TILE_SIZE + tx;

    // Accumulate result
    float value = 0.0f;

    // Loop over tiles of K dimension
    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Shared memory for A and B tiles
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // Load tile of A into shared memory
        if (row < M && (m * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + m * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        if (col < N && (m * TILE_SIZE + ty) < K) {
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().device(A.device()).dtype(A.dtype());
    auto C = torch::zeros({M, N}, options);

    const int TILE_SIZE = 16;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K);

    return C;
}
"""

matmul_cpp_code = r"""
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix multiplication
matmul_extension = load_inline(
    name='matmul_extension',
    cpp_sources=matmul_cpp_code,
    cuda_sources=matmul_kernel_code,
    functions=['matmul_cuda'],
    verbose=False,
    extra_cuda_cflags=['--use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_extension.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda(A.contiguous(), B.contiguous())