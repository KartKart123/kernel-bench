import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA code for upper triangular matrix multiplication
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N && i <= j) {
        float sum = 0.0f;
        for (int k = i; k <= j; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 block_size(16, 16);
    dim3 num_blocks((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    upper_triangular_matmul_kernel<<<num_blocks, block_size>>>(A_data, B_data, C_data, N);

    return C;
}
"""

cpp_source = """
torch::Tensor upper_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
upper_triangular_matmul = load_inline(
    name='upper_triangular_matmul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['upper_triangular_matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.upper_triangular_matmul = upper_triangular_matmul

    def forward(self, A, B):
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()
        return self.upper_triangular_matmul.upper_triangular_matmul_cuda(A, B)

N = 4096

def get_inputs():
    A = torch.triu(torch.randn(N, N)).cuda()
    B = torch.triu(torch.randn(N, N)).cuda()
    return [A, B]

def get_init_inputs():
    return []