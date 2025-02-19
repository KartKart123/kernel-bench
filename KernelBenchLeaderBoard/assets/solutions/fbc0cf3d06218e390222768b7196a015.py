import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = """
#include <torch/extension.h>

__global__ void softsign_kernel(const float* x, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = x[i] / (1.0f + fabsf(x[i]));
  }
}

torch::Tensor softsign_cuda(torch::Tensor x) {
  auto size = x.numel();
  auto out = torch::zeros_like(x);

  const int threads_per_block = 256;
  const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  softsign_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

  return out;
}
"""

softsign_cpp_source = "torch::Tensor softsign_cuda(torch::Tensor x);"

softsign_module = load_inline(
    name='softsign_cuda',
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=['softsign_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign = softsign_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign.softsign_cuda(x)