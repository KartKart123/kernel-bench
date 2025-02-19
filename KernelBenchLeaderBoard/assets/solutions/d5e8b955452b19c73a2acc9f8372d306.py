import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            const int M, const int N, const int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within block
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Each thread computes one element of the block sub-matrix
    float sum = 0.0f;
    
    // Loop over all sub-matrices of A and B required to compute block sub-matrix
    for (int m = 0; m < K; m += 16) {
        // Load sub-matrices from global memory to shared memory
        __shared__ float As[16][16];
        __shared__ float Bs[16][16];
        
        if ((blockRow * 16 + row < M) && (m + col < K))
            As[row][col] = A[(blockRow * 16 + row) * K + m + col];
        else
            As[row][col] = 0.0f;
            
        if ((m + row < K) && (blockCol * 16 + col < N))
            Bs[row][col] = B[(m + row) * N + blockCol * 16 + col];
        else
            Bs[row][col] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < 16; k++)
            sum += As[row][k] * Bs[k][col];
            
        __syncthreads();
    }
    
    // Write result to global memory
    if ((blockRow * 16 + row < M) && (blockCol * 16 + col < N))
        C[(blockRow * 16 + row) * N + blockCol * 16 + col] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name='matmul_cuda',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=['matmul_cuda'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A.cuda(), B.cuda())