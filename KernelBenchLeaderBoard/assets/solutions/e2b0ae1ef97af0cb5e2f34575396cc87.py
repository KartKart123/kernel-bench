import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triu_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triu_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        for (int k = row; k <= col; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    triu_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
"""

triu_matmul_cpp_source = """
torch::Tensor triu_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

triu_matmul = load_inline(
    name='triu_matmul',
    cpp_sources=triu_matmul_cpp_source,
    cuda_sources=triu_matmul_source,
    functions=['triu_matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.triu_matmul = triu_matmul

    def forward(self, A, B):
        return self.triu_matmul.triu_matmul_cuda(A.cuda(), B.cuda())