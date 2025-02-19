import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tensor-matrix multiplication
tensor_matrix_multiply_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tensor_matrix_multiply_kernel(
    const float* __restrict__ A,    // (b, i, j, l)
    const float* __restrict__ B,    // (l, k)
    float* __restrict__ C,          // (b, i, j, k)
    int b, int i, int j, int l, int k)
{
    int total_batches = b * i * j;
    int batch_idx = blockIdx.x;

    if (batch_idx >= total_batches) return;

    int b_idx = batch_idx / (i * j);
    int ij_idx = batch_idx % (i * j);
    int i_idx = ij_idx / j;
    int j_idx = ij_idx % j;

    int k_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (k_idx >= k) return;

    // Index pointers
    int A_index = ((b_idx * i + i_idx) * j + j_idx) * l;
    int C_index = ((b_idx * i + i_idx) * j + j_idx) * k + k_idx;

    float sum = 0.0f;

    for (int l_idx = 0; l_idx < l; ++l_idx)
    {
        float a_val = A[A_index + l_idx];          // A[b_idx, i_idx, j_idx, l_idx]
        float b_val = B[l_idx * k + k_idx];        // B[l_idx, k_idx]
        sum += a_val * b_val;
    }

    C[C_index] = sum;
}

torch::Tensor tensor_matrix_multiply_cuda(torch::Tensor A, torch::Tensor B)
{
    int b = A.size(0);
    int i = A.size(1);
    int j = A.size(2);
    int l = A.size(3);
    int k = B.size(1);

    TORCH_CHECK(B.size(0) == l, "B.size(0) must be equal to A.size(3)");

    // Ensure A and B are contiguous and on the correct device
    A = A.contiguous();
    B = B.contiguous();

    auto C = torch::zeros({b, i, j, k}, A.options());

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    int total_batches = b * i * j;

    int threads_per_block = 256;
    int blocks_per_k = (k + threads_per_block - 1) / threads_per_block;

    // Use a 2D grid: x dimension is total_batches, y dimension is blocks per k
    dim3 grid_dim(total_batches, blocks_per_k);
    dim3 block_dim(threads_per_block);

    // Launch kernel
    tensor_matrix_multiply_kernel<<<grid_dim, block_dim>>>(
        A_ptr, B_ptr, C_ptr, b, i, j, l, k);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();

    return C;
}
"""

tensor_matrix_multiply_cpp_source = """
torch::Tensor tensor_matrix_multiply_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code
tensor_matrix_multiply = load_inline(
    name='tensor_matrix_multiply',
    cpp_sources=[tensor_matrix_multiply_cpp_source],
    cuda_sources=[tensor_matrix_multiply_source],
    functions=['tensor_matrix_multiply_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matrix_multiply = tensor_matrix_multiply

    def forward(self, A, B):
        return self.tensor_matrix_multiply.tensor_matrix_multiply_cuda(A, B)