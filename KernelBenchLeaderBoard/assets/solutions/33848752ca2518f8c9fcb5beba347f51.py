import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_norm_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void l2_norm_kernel(const float* x, float* out, int batch_size, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size) {
    float norm = 0.0f;
    for (int j = 0; j < dim; ++j) {
      norm += x[i * dim + j] * x[i * dim + j];
    }
    norm = sqrtf(norm);
    for (int j = 0; j < dim; ++j) {
      out[i * dim + j] = x[i * dim + j] / norm;
    }
  }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
  auto batch_size = x.size(0);
  auto dim = x.size(1);
  auto out = torch::zeros_like(x);

  const int block_size = 256;
  const int num_blocks = (batch_size + block_size - 1) / block_size;

  l2_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

  return out;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor x);"

l2_norm = load_inline(
    name='l2_norm',
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=['l2_norm_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_norm.l2_norm_cuda(x)