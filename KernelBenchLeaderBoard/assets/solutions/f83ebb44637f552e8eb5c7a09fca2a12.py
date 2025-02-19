import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication of symmetric matrices
matmul_symmetric_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_symmetric_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_symmetric_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block_size(16, 16);
    dim3 num_blocks((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    matmul_symmetric_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_symmetric_cpp_source = "torch::Tensor matmul_symmetric_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication of symmetric matrices
matmul_symmetric = load_inline(
    name='matmul_symmetric',
    cpp_sources=matmul_symmetric_cpp_source,
    cuda_sources=matmul_symmetric_source,
    functions=['matmul_symmetric_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_symmetric = matmul_symmetric

    def forward(self, A, B):
        return self.matmul_symmetric.matmul_symmetric_cuda(A, B)