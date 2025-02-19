import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cmath>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size * dim) {
    int b = i / dim;
    int d = i % dim;
    float max_val = input[b * dim + 0];
    for (int j = 1; j < dim; ++j) {
      max_val = fmaxf(max_val, input[b * dim + j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < dim; ++j) {
      sum += expf(input[b * dim + j] - max_val);
    }
    output[i] = expf(input[b * dim + d] - max_val) / sum;
  }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
  int batch_size = input.size(0);
  int dim = input.size(1);
  auto output = torch::zeros_like(input);

  const int block_size = 256;
  const int num_blocks = (batch_size * dim + block_size - 1) / block_size;

  softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

  return output;
}
"""

softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor input);"

softmax_op = load_inline(
    name='softmax_op',
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=['softmax_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax.softmax_cuda(x)