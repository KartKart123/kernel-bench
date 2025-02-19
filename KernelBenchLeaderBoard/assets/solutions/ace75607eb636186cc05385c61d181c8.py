import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and function for matrix-vector multiplication
matvec_mul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matvec_mul_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x;  // Each block computes one row
    int tid = threadIdx.x;

    extern __shared__ float shared_sum[];  // Shared memory for partial sums

    // Each thread computes partial sum over parts of K
    float sum = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        sum += A[row * K + i] * B[i];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[row] = shared_sum[0];
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check device
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    // Check dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.size(0) == K && B.size(1) == 1, "B must have size K x 1");

    // Allocate output tensor
    auto C = torch::empty({M, 1}, A.options());

    // Launch kernel
    int threads = 256;
    int blocks = M;
    size_t shared_mem_size = threads * sizeof(float);

    matvec_mul_kernel<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

matvec_mul_cpp_source = """
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix-vector multiplication
matvec_mul = load_inline(
    name='matvec_mul',
    cpp_sources=matvec_mul_cpp_source,
    cuda_sources=matvec_mul_source,
    functions=['matvec_mul_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec_mul = matvec_mul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec_mul.matvec_mul_cuda(A, B)