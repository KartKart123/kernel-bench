import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

swish_cuda_source = """
#include <torch/extension.h>

__global__ void swish_kernel(const float* x, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = x[i] * (1.0f / (1.0f + expf(-x[i])));
  }
}

torch::Tensor swish_cuda(torch::Tensor x) {
  auto size = x.numel();
  auto out = torch::zeros_like(x);

  const int threads_per_block = 256;
  const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  swish_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

  return out;
}
"""

swish_cpp_source = "torch::Tensor swish_cuda(torch::Tensor x);"

swish_cuda_module = load_inline(
    name="swish_cuda",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_cuda_source,
    functions=["swish_cuda"],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish = swish_cuda_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish.swish_cuda(x)