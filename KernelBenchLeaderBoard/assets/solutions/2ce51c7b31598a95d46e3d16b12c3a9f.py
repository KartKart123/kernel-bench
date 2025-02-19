import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for diagonal matrix multiplication
diag_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* out, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < N && idy < M) {
        out[idx * M + idy] = A[idx] * B[idx * M + idy];
    }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto N = A.numel();
    auto M = B.size(1);
    auto out = torch::zeros_like(B);

    const int block_size = 16;
    const dim3 num_blocks((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);

    diag_matmul_kernel<<<num_blocks, dim3(block_size, block_size)>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M);

    return out;
}
"""

diag_matmul_cpp_source = (
    "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for diagonal matrix multiplication
diag_matmul = load_inline(
    name="diag_matmul",
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=["diag_matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)