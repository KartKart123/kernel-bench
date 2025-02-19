import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the C++ function signature
batched_matmul_cpp_source = '''
torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);
'''

# Define the CUDA kernel and the interface function
batched_matmul_cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void batched_matrix_multiply_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int batch_size,
    int M, int K, int N) {

    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension

    if (batch < batch_size && row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a_element = A[batch * M * K + row * K + i];
            float b_element = B[batch * K * N + i * N + col];
            value += a_element * b_element;
        }
        C[batch * M * N + row * N + col] = value;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check input tensors are on the same device
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be CUDA tensors.");
    }

    // Ensure input tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    // Ensure that the batch sizes and inner dimensions match
    if (batch_size != B.size(0) || K != B.size(1)) {
        throw std::invalid_argument("Input tensor dimensions do not match for batched matrix multiplication.");
    }

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::zeros({batch_size, M, N}, options);

    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE,
                batch_size);

    batched_matrix_multiply_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N);

    return C;
}
'''

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name='batched_matmul',
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_cuda_source,
    functions=['batched_matmul_cuda'],
    extra_cuda_cflags=['--restrict'],
    verbose=True
)

# Define the new model that uses the custom CUDA kernel
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A, B):
        return self.batched_matmul.batched_matmul_cuda(A, B)