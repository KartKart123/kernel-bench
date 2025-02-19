import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void lower_triangular_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N && i >= j) {
        float sum = 0.0f;
        for (int k = j; k <= i; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");
    int N = A.size(0);

    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same device");

    if (!A.is_contiguous()) A = A.contiguous();
    if (!B.is_contiguous()) B = B.contiguous();

    auto C = torch::zeros({N, N}, A.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    cudaSetDevice(A.device().index());

    lower_triangular_matmul_kernel<<<blocks, threads>>>(
        A_data, B_data, C_data, N
    );

    return C;
}
"""

cpp_source = """
torch::Tensor lower_triangular_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

lower_triangular_matmul = load_inline(
    name="lower_triangular_matmul",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["lower_triangular_matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.lower_triangular_matmul = lower_triangular_matmul.lower_triangular_matmul_cuda

    def forward(self, A, B):
        return self.lower_triangular_matmul(A, B)