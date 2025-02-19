import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batched matrix multiplication
batched_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int batch_size, int m, int k, int n) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float value = 0.0;
        for (int i = 0; i < k; ++i) {
            value += A[batch_idx * m * k + row * k + i] * B[batch_idx * k * n + i * n + col];
        }
        C[batch_idx * m * n + row * n + col] = value;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto batch_size = A.size(0);
    auto m = A.size(1);
    auto k = A.size(2);
    auto n = B.size(2);
    auto C = torch::zeros({batch_size, m, n}, A.options());

    const dim3 block_size(16, 16);
    const dim3 num_blocks((n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y, batch_size);

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
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)