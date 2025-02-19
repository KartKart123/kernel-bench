import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D tensor-matrix multiplication
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tensor3d_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int M, int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * M * L;

    if (idx < total_size) {
        int n = idx / (M * L);
        int m = (idx % (M * L)) / L;
        int l = idx % L;

        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a_val = A[((n * M + m) * K) + k];
            float b_val = B[k * L + l];
            sum += a_val * b_val;
        }
        C[((n * M + m) * L) + l] = sum;
    }
}

torch::Tensor tensor3d_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    A = A.contiguous();
    B = B.contiguous();

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    auto C = torch::zeros({N, M, L}, A.options());

    int total_size = N * M * L;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    tensor3d_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M, K, L);

    return C;
}
"""

cpp_source = """
torch::Tensor tensor3d_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for the custom tensor-matrix multiplication
tensor3d_matmul = load_inline(
    name='tensor3d_matmul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['tensor3d_matmul_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor3d_matmul = tensor3d_matmul

    def forward(self, A, B):
        return self.tensor3d_matmul.tensor3d_matmul_cuda(A, B)

N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = torch.randn(N, M, K).cuda()
    B = torch.randn(K, L).cuda()
    return [A, B]

def get_init_inputs():
    return []