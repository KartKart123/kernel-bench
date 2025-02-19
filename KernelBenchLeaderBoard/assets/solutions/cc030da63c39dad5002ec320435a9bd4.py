import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

diag_matmul_source = """
#include <torch/extension.h>

__global__ void diag_matmul_kernel(const float* A, const float* B, float* C, int N, int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    for (int j = 0; j < M; ++j) {
      C[i * M + j] = A[i] * B[i * M + j];
    }
  }
}

torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B) {
  int N = A.size(0);
  int M = B.size(1);
  auto C = torch::zeros({N, M}, A.options());

  const int block_size = 256;
  const int num_blocks = (N + block_size - 1) / block_size;

  diag_matmul_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

  return C;
}
"""

diag_matmul_cpp_source = "torch::Tensor diag_matmul_cuda(torch::Tensor A, torch::Tensor B);"

diag_matmul = load_inline(
    name='diag_matmul',
    cpp_sources=diag_matmul_cpp_source,
    cuda_sources=diag_matmul_source,
    functions=['diag_matmul_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.diag_matmul = diag_matmul

    def forward(self, A, B):
        return self.diag_matmul.diag_matmul_cuda(A, B)