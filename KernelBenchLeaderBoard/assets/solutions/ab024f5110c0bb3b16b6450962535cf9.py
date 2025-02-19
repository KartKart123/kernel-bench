import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

leaky_relu_source = """
#include <torch/extension.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int size, float negative_slope) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = (input[i] > 0) ? input[i] : input[i] * negative_slope;
  }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope) {
  auto size = input.numel();
  auto output = torch::zeros_like(input);

  const int threads_per_block = 256;
  const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  leaky_relu_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, negative_slope);

  return output;
}
"""

leaky_relu_cpp_source = "torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope);"

leaky_relu_module = load_inline(
    name='leaky_relu_cuda',
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=['leaky_relu_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leaky_relu = leaky_relu_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu.leaky_relu_cuda(x, self.negative_slope)