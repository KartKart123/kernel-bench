import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_source = """
#include <torch/extension.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = (input[i] > 0) ? input[i] : 0;
  }
}

torch::Tensor relu_cuda(torch::Tensor input) {
  auto size = input.numel();
  auto output = torch::zeros_like(input);

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

  return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

relu_module = load_inline(
    name="relu_module",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu = relu_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.relu_cuda(x)