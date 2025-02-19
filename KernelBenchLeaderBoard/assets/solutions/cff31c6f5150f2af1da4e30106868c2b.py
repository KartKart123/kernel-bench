import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int batch_size, int m, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < batch_size * m * n; i += stride) {
        int b = i / (m * n);
        int row = (i % (m * n)) / n;
        int col = (i % (m * n)) % n;

        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            sum += A[b * m * k + row * k + j] * B[b * k * n + j * n + col];
        }
        C[b * m * n + row * n + col] = sum;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto batch_size = A.size(0);
    auto m = A.size(1);
    auto k = A.size(2);
    auto n = B.size(2);

    auto C = torch::zeros({batch_size, m, n}, A.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * m * n + block_size - 1) / block_size;

    batched_matmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, m, k, n);

    return C;
}
"""

batched_matmul_cpp_source = "torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name='batched_matmul',
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_source,
    functions=['batched_matmul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)