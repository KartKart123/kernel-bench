import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code
matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int64_t M, int64_t N, int64_t K) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Row and column indices of the element to compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // M dimension
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // N dimension
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load elements into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    // Write the result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t K_B = B.size(0);
    int64_t N = B.size(1);

    TORCH_CHECK(K == K_B, "A's columns must equal B's rows");

    // Allocate output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Check if tensors are contiguous and on CUDA
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA device");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");

    // Launch kernel
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1)/TILE_SIZE);

    matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K);

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name='matmul',
    cpp_sources=[matmul_cpp_source],
    cuda_sources=[matmul_cuda_source],
    functions=['matmul_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)