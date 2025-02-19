import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
matvec_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matvec_mul_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k];
        }
        C[row] = sum;
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    auto C = torch::zeros({M, 1}, A.options());

    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;

    matvec_mul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

matvec_mul_cpp_source = "torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix-vector multiplication
matvec_mul = load_inline(
    name='matvec_mul',
    cpp_sources=matvec_mul_cpp_source,
    cuda_sources=matvec_mul_source,
    functions=['matvec_mul_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec_mul = matvec_mul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec_mul.matvec_mul_cuda(A, B)