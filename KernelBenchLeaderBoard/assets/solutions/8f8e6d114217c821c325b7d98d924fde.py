import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tensor_matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D tensor-matrix multiplication
__global__ void tensor_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float* __restrict__ C,
    const int N, const int M, const int K, const int L) {

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Block size
    const int BLOCK_SIZE = 16;

    // Shared memory for tiling
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Global indices
    const int row = by * BLOCK_SIZE + ty;
    const int col = bx * BLOCK_SIZE + tx;
    const int batch = bz;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tile from A into shared memory
        if (row < M && (t * BLOCK_SIZE + tx) < K && batch < N) {
            As[ty][tx] = A[batch * M * K + row * K + t * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        if ((t * BLOCK_SIZE + ty) < K && col < L) {
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * L + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < L && batch < N) {
        C[batch * M * L + row * L + col] = sum;
    }
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    auto C = torch::zeros({N, M, L}, A.options());

    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (L + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
        N
    );

    tensor_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M, K, L
    );

    return C;
}
"""

tensor_matmul_cpp_source = """
torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

tensor_matmul = load_inline(
    name='tensor_matmul',
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_cuda_source,
    functions=['tensor_matmul_cuda'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matmul_cuda(A.cuda(), B.cuda())