import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication C = A^T * B
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            int M, int K, int N) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within block
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Each thread computes one element of the block sub-matrix
    float sum = 0.0f;
    
    // Loop over all sub-matrices needed to compute this block
    for (int m = 0; m < (K + 31) / 32; m++) {
        // Shared memory for the sub-matrices of A and B
        __shared__ float As[32][32];
        __shared__ float Bs[32][32];
        
        // Load the matrices from global memory to shared memory
        if (m * 32 + col < K && blockRow * 32 + row < M)
            As[row][col] = A[(m * 32 + col) * M + blockRow * 32 + row];
        else
            As[row][col] = 0.0f;
            
        if (m * 32 + row < K && blockCol * 32 + col < N)
            Bs[row][col] = B[(m * 32 + row) * N + blockCol * 32 + col];
        else
            Bs[row][col] = 0.0f;
            
        __syncthreads();
        
        // Multiply the two matrices together
        for (int k = 0; k < 32; k++)
            sum += As[row][k] * Bs[k][col];
            
        __syncthreads();
    }
    
    // Write the block sub-matrix to global memory
    if (blockRow * 32 + row < M && blockCol * 32 + col < N)
        C[(blockRow * 32 + row) * N + blockCol * 32 + col] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(1);
    const int K = A.size(0);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (M + 31) / 32);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_matmul = load_inline(
    name='matmul',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=['matmul_cuda'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_matmul = custom_matmul
        
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.custom_matmul.matmul_cuda(A, B)