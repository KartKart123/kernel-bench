import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication of symmetric matrices
symmetric_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void symmetric_matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros_like(A);

    const int block_size = 16;
    const int num_blocks_x = (N + block_size - 1) / block_size;
    const int num_blocks_y = (N + block_size - 1) / block_size;

    symmetric_matmul_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

symmetric_matmul_cpp_source = "torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication of symmetric matrices
symmetric_matmul = load_inline(
    name='symmetric_matmul',
    cpp_sources=symmetric_matmul_cpp_source,
    cuda_sources=symmetric_matmul_source,
    functions=['symmetric_matmul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.symmetric_matmul = symmetric_matmul

    def forward(self, A, B):
        return self.symmetric_matmul.symmetric_matmul_cuda(A, B)