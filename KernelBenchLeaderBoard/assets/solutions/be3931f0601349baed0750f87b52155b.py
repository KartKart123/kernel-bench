import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

softplus_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void softplus_kernel(const float* x, float* out, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = logf(1.0f + expf(x[i]));
  }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
  auto size = x.numel();
  auto out = torch::zeros_like(x);

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;

  softplus_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
  return out;
}
"""

softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x);"

softplus_module = load_inline(
    name='softplus_cuda',
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=['softplus_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus = softplus_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus.softplus_cuda(x)