import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
matmul_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_3d_kernel(const float* A, const float* B, float* out, int N, int M, int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < N && idy < M && idz < L) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[idx * M * K + idy * K + k] * B[k * L + idz];
        }
        out[idx * M * L + idy * L + idz] = sum;
    }
}

torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    auto out = torch::zeros({N, M, L}, torch::TensorOptions().device(A.device()));

    const int block_size_x = 8;
    const int block_size_y = 8;
    const int block_size_z = 8;
    const int num_blocks_x = (N + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (M + block_size_y - 1) / block_size_y;
    const int num_blocks_z = (L + block_size_z - 1) / block_size_z;

    dim3 block(block_size_x, block_size_y, block_size_z);
    dim3 grid(num_blocks_x, num_blocks_y, num_blocks_z);

    matmul_3d_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M, K, L);

    return out;
}
"""

matmul_3d_cpp_source = (
    "torch::Tensor matmul_3d_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for 3D tensor-matrix multiplication
matmul_3d = load_inline(
    name="matmul_3d",
    cpp_sources=matmul_3d_cpp_source,
    cuda_sources=matmul_3d_source,
    functions=["matmul_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul_3d = matmul_3d

    def forward(self, A, B):
        return self.matmul_3d.matmul_3d_cuda(A, B)