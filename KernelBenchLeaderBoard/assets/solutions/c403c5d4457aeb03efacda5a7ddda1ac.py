import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

fused_matmul_sub_mul_relu_source = """
#include <torch/extension.h>

__global__ void fused_matmul_sub_mul_relu_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_features, int out_features, float subtract_value, float multiply_value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < batch_size * out_features) {
    int batch_index = i / out_features;
    int output_index = i % out_features;
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
      sum += input[batch_index * in_features + k] * weight[output_index * in_features + k];
    }
    sum += bias[output_index];
    sum -= subtract_value;
    sum *= multiply_value;
    output[i] = fmaxf(0.0f, sum);
  }
}


torch::Tensor fused_matmul_sub_mul_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float subtract_value, float multiply_value) {
  int batch_size = input.size(0);
  int in_features = input.size(1);
  int out_features = weight.size(0);

  torch::Tensor output = torch::zeros({batch_size, out_features}, input.options());

  const int block_size = 256;
  const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

  fused_matmul_sub_mul_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_features, out_features, subtract_value, multiply_value);

  return output;
}
"""

fused_matmul_sub_mul_relu_cpp_source = "torch::Tensor fused_matmul_sub_mul_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float subtract_value, float multiply_value);"

fused_kernel = load_inline(
    name='fused_kernel',
    cpp_sources=fused_matmul_sub_mul_relu_cpp_source,
    cuda_sources=fused_matmul_sub_mul_relu_source,
    functions=['fused_matmul_sub_mul_relu_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation using a fused CUDA kernel.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_kernel = fused_kernel

    def forward(self, x):
        weight = self.linear.weight
        bias = self.linear.bias
        x = self.fused_kernel.fused_matmul_sub_mul_relu_cuda(x, weight, bias, self.subtract_value, self.multiply_value)
        return x