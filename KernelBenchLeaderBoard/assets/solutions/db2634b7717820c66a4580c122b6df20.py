import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of the output matrix C to compute
    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;

    float Cvalue = 0.0;

    // Loop over the tiles of K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1)/TILE_SIZE; ++t) {

        // Shared memory for tiles
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // Load elements into shared memory
        if (Row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[Row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0;

        if ((t * TILE_SIZE + ty) < K && Col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + Col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    // Write the result to C
    if (Row < M && Col < N)
        C[Row * N + Col] = Cvalue;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get matrix dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE -1)/TILE_SIZE);

    // Ensure the tensors are on the same device and are contiguous
    A = A.contiguous();
    B = B.contiguous();

    // Launch the kernel
    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

matmul_cuda_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_cuda = load_inline(
    name='matmul_cuda',
    cpp_sources=matmul_cuda_cpp_source,
    cuda_sources=matmul_cuda_kernel,
    functions=['matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A, B):
        return self.matmul_cuda.matmul_cuda(A, B)