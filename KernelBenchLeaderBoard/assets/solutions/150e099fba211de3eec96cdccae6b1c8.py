import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Matrix multiplication kernel optimized for matrix-vector multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            const int M, const int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        
        // Each thread computes one element of the result vector
        for (int k = 0; k < K; k += 4) {
            // Manual unrolling for better instruction-level parallelism
            float4 a = *reinterpret_cast<const float4*>(&A[row * K + k]);
            float4 b = *reinterpret_cast<const float4*>(&B[k]);
            
            sum += a.x * b.x;
            sum += a.y * b.y;
            sum += a.z * b.z;
            sum += a.w * b.w;
        }
        
        C[row] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    
    auto C = torch::zeros({M, 1}, A.options());
    
    const int threads = 256;
    const int blocks = (M + threads - 1) / threads;
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K
    );
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

custom_matmul = load_inline(
    name='matmul_cuda',
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