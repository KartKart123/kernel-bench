import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x < N && idx_y < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[idx_y * N + i] * B[i * N + idx_x];
        }
        C[idx_y * N + idx_x] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.size(0);
    auto C = torch::zeros_like(A);

    const int block_size = 16;
    const dim3 num_blocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    const dim3 block(block_size, block_size);

    matmul_kernel<<<num_blocks, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}
"""

matmul_cpp_source = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix multiplication
matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul = matmul

    def forward(self, A, B):
        return self.matmul.matmul_cuda(A, B)