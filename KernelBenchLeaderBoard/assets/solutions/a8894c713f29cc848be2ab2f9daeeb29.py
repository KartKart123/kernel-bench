import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA code
matmul_cuda_source = """
#include <torch/extension.h>

#define TILE_WIDTH 16

__global__ void matmul_kernel(const float* A, const float* B_T, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // M dimension
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // N dimension

    float Cvalue = 0.0f;

    // Loop over the tiles of K dimension
    for (int t = 0; t < ( (K + TILE_WIDTH - 1) / TILE_WIDTH ); ++t) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load tile of B_T into shared memory
        if (col < N && t * TILE_WIDTH + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B_T[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial product
        for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N)
        C[row * N + col] = Cvalue;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B_T, int M, int N, int K) {
    auto C = torch::zeros({M, N}, A.options());

    dim3 block_size(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_size((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B_T.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K
    );

    return C;
}
"""

# Declare the function prototype
matmul_cuda_cpp_source = '''
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B_T, int M, int N, int K);
'''

# Load the inline CUDA code
matmul = load_inline(
    name='matmul_cuda',
    cpp_sources=matmul_cuda_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=['matmul_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        N = B.shape[0]

        # Ensure tensors are contiguous and on the same device
        A = A.contiguous().cuda()
        B = B.contiguous().cuda()

        # Transpose B to get B_T
        B_T = B.t().contiguous()

        # Call the custom CUDA matmul
        return self.matmul.matmul_cuda(A, B_T, M, N, K)

M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(N, K)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed