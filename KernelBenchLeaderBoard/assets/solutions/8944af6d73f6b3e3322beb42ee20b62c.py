import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define M, K, N
M = 1024
K = 4096
N = 2048

CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 16

__global__ void MatMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    float Cvalue = 0;

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int m = 0; m < numTiles; ++m) {

        int rowA = blockRow * BLOCK_SIZE + threadRow;
        int colA = m * BLOCK_SIZE + threadCol;

        if (rowA < M && colA < K) {
            Asub[threadRow][threadCol] = A[rowA * K + colA];
        } else {
            Asub[threadRow][threadCol] = 0.0f;
        }

        int rowB = m * BLOCK_SIZE + threadRow;
        int colB = blockCol * BLOCK_SIZE + threadCol;

        if (rowB < K && colB < N) {
            Bsub[threadRow][threadCol] = B[rowB * N + colB];
        } else {
            Bsub[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += Asub[threadRow][e] * Bsub[e][threadCol];
        }

        __syncthreads();
    }

    int rowC = blockRow * BLOCK_SIZE + threadRow;
    int colC = blockCol * BLOCK_SIZE + threadCol;

    if (rowC < M && colC < N) {
        C[rowC * N + colC] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {

    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int K_B = B.size(0);
    int N = B.size(1);

    TORCH_CHECK(K == K_B, "Matrices have incompatible dimensions");

    auto C = torch::zeros({M, N}, A.options());

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    MatMulKernel<<<dimGrid, dimBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

CPP_SOURCE = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Load the inline extension
matmul_cuda = load_inline(
    name='matmul_cuda',
    cpp_sources=CPP_SOURCE,
    cuda_sources=CUDA_SOURCE,
    functions=['matmul_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A, B):
        A_t = A.T.contiguous()
        B_t = B.T.contiguous()
        return self.matmul_cuda.matmul_cuda(A_t, B_t)

def get_inputs():
    A = torch.randn(K, M, device='cuda')
    B = torch.randn(N, K, device='cuda')
    return [A, B]

def get_init_inputs():
    return []