import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix-vector multiplication
matvecmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matvecmul_kernel(const float* A, const float* B, float* out, int M, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[idx * K + k] * B[k];
        }
        out[idx] = sum;
    }
}

torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto out = torch::zeros({M}, A.options());

    const int block_size = 256;
    const int num_blocks = (M + block_size - 1) / block_size;

    matvecmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), M, K);

    return out.view({M, 1});
}
"""

matvecmul_cpp_source = (
    "torch::Tensor matvecmul_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for matrix-vector multiplication
matvecmul = load_inline(
    name="matvecmul",
    cpp_sources=matvecmul_cpp_source,
    cuda_sources=matvecmul_source,
    functions=["matvecmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matvecmul = matvecmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvecmul.matvecmul_cuda(A, B)