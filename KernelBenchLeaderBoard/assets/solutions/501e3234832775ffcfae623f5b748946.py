import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

symmetric_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized kernel for symmetric matrix multiplication
__global__ void symmetric_matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Block dimensions
    const int BLOCK_SIZE = 32;
    
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Index of first sub-matrix of A processed by block
    int aBegin = N * BLOCK_SIZE * by;
    // Index of last sub-matrix of A processed by block
    int aEnd = aBegin + N - 1;
    // Step size used to iterate through sub-matrices of A
    int aStep = BLOCK_SIZE;
    // Index of first sub-matrix of B processed by block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through sub-matrices of B
    int bStep = BLOCK_SIZE * N;
    
    float sum = 0.0f;
    
    // Loop over all sub-matrices of A and B required for block
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load tiles into shared memory
        if (a + N * ty + tx < N * N && ty < BLOCK_SIZE && tx < BLOCK_SIZE) {
            As[ty][tx] = A[a + N * ty + tx];
            Bs[ty][tx] = B[b + N * ty + tx];
        } else {
            As[ty][tx] = 0.0f;
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (by * BLOCK_SIZE + ty < N && bx * BLOCK_SIZE + tx < N) {
        C[(by * BLOCK_SIZE + ty) * N + bx * BLOCK_SIZE + tx] = sum;
    }
}

torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    
    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    symmetric_matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

symmetric_matmul_cpp_source = """
torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

symmetric_matmul = load_inline(
    name='symmetric_matmul',
    cpp_sources=symmetric_matmul_cpp_source,
    cuda_sources=symmetric_matmul_source,
    functions=['symmetric_matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.symmetric_matmul = symmetric_matmul
        
    def forward(self, A, B):
        return self.symmetric_matmul.symmetric_matmul_cuda(A, B)