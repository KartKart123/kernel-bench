import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_source = """
#include <torch/extension.h>

__global__ void hardtanh_kernel(const float* x, float* y, int size, float min_val, float max_val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    y[i] = fmaxf(min_val, fminf(x[i], max_val));
  }
}

torch::Tensor hardtanh_cuda(torch::Tensor x, float min_val, float max_val) {
  auto size = x.numel();
  auto y = torch::empty_like(x);

  const int threads_per_block = 256;
  const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  hardtanh_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), size, min_val, max_val);

  return y;
}
"""

hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor x, float min_val, float max_val);"

hardtanh_cuda = load_inline(
    name='hardtanh_cuda',
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=['hardtanh_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardtanh = hardtanh_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh.hardtanh_cuda(x, -1.0, 1.0)