import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for lower triangular matrix multiplication
tril_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tril_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && col <= row) {
        float sum = 0.0f;
        for (int k = 0; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const int block_size = 16;
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    tril_matmul_kernel<<<dim_grid, dim_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

tril_matmul_cpp_source = "torch::Tensor tril_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for lower triangular matrix multiplication
tril_matmul = load_inline(
    name='tril_matmul',
    cpp_sources=tril_matmul_cpp_source,
    cuda_sources=tril_matmul_source,
    functions=['tril_matmul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tril_matmul = tril_matmul
    
    def forward(self, A, B):
        return self.tril_matmul.tril_matmul_cuda(A, B)