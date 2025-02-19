import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int batch_size, int m, int k, int n) {
    int batch_idx = blockIdx.z;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && row_idx < m && col_idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[batch_idx * m * k + row_idx * k + i] * B[batch_idx * k * n + i * n + col_idx];
        }
        C[batch_idx * m * n + row_idx * n + col_idx] = sum;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);

    auto C = torch::zeros({batch_size, m, n}, torch::TensorOptions().device(A.device()));

    const int block_size_x = 16;
    const int block_size_y = 16;
    const int num_blocks_x = (n + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (m + block_size_y - 1) / block_size_y;

    dim3 block(block_size_x, block_size_y);
    dim3 grid(num_blocks_x, num_blocks_y, batch_size);

    batched_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, m, k, n);

    return C;
}
"""

batched_matmul_cpp_source = (
    "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name="batched_matmul",
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=["batched_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)