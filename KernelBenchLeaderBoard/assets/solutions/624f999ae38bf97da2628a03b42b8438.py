import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

gemm_relu_source = """
#include <torch/extension.h>

__global__ void gemm_relu_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_features, int out_features) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size * out_features) {
    int batch_index = i / out_features;
    int out_index = i % out_features;
    float sum = 0.0f;
    for (int j = 0; j < in_features; ++j) {
      sum += input[batch_index * in_features + j] * weight[out_index * in_features + j];
    }
    sum += bias[out_index];
    output[i] = fmaxf(0.0f, sum);
  }
}

torch::Tensor gemm_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
  int batch_size = input.size(0);
  int in_features = input.size(1);
  int out_features = weight.size(0);

  torch::Tensor output = torch::zeros({batch_size, out_features}, input.options());

  const int block_size = 256;
  const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

  gemm_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_features, out_features);

  return output;
}
"""

gemm_relu_cpp_source = "torch::Tensor gemm_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

gemm_relu = load_inline(
    name='gemm_relu',
    cpp_sources=gemm_relu_cpp_source,
    cuda_sources=gemm_relu_source,
    functions=['gemm_relu_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.gemm_relu = gemm_relu

    def forward(self, x):
        weight = self.gemm.weight
        bias = self.bias
        x = self.gemm_relu.gemm_relu_cuda(x, weight, bias)
        return x