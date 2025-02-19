import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(
    const float* diag,
    const float* mat,
    float* out,
    const int N,
    const int M) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        out[row * M + col] = diag[row] * mat[row * M + col];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor mat) {
    const int N = diag.size(0);
    const int M = mat.size(1);
    
    auto out = torch::zeros({N, M}, mat.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks((M + threads.x - 1) / threads.x,
                     (N + threads.y - 1) / threads.y);
                     
    diag_matmul_kernel<<<blocks, threads>>>(
        diag.data_ptr<float>(),
        mat.data_ptr<float>(),
        out.data_ptr<float>(),
        N, M);
        
    return out;
}
"""

diag_matmul_cpp_source = """
torch::Tensor diag_matmul_cuda(torch::Tensor diag, torch::Tensor mat);
"""

diag_matmul = load_inline(
    name='diag_matmul',
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_cuda_source,
    functions=['diag_matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul
        
    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A.cuda(), B.cuda())