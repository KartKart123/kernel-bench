import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_source = """
#include <torch/extension.h>

__global__ void tanh_kernel(const float* x, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = tanhf(x[i]);
  }
}

torch::Tensor tanh_cuda(torch::Tensor x) {
  auto size = x.numel();
  auto out = torch::zeros_like(x);

  const int threads_per_block = 256;
  const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  tanh_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

  return out;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor x);"

tanh_op = load_inline(
    name='tanh_op',
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=['tanh_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_op = tanh_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_op.tanh_cuda(x)