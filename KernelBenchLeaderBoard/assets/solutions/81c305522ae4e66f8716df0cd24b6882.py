import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

elu_source = """
#include <torch/extension.h>

__global__ void elu_kernel(const float* x, float* out, int size, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    if (x[i] > 0) {
      out[i] = x[i];
    } else {
      out[i] = alpha * (expf(x[i]) - 1.0f);
    }
  }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
  auto size = x.numel();
  auto out = torch::zeros_like(x);

  const int threads_per_block = 256;
  const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  elu_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, alpha);

  return out;
}
"""

elu_cpp_source = "torch::Tensor elu_cuda(torch::Tensor x, float alpha);"

elu_module = load_inline(
    name='elu_cuda',
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_source,
    functions=['elu_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    """
    Simple model that performs an ELU activation using a custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda = elu_module.elu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch::Tensor: Output tensor with ELU applied, same shape as input.
        """
        return self.elu_cuda(x, self.alpha)