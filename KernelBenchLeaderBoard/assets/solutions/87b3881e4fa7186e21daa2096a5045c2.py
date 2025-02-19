import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matmul + divide + GELU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int input_size, int output_size, float divisor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * output_size) {
        int b = i / output_size;
        int o = i % output_size;
        float sum = 0.0f;
        for (int k = 0; k < input_size; ++k) {
            sum += input[b * input_size + k] * weight[o * input_size + k];
        }
        sum += bias[o];
        sum /= divisor;
        output[i] = sum * 0.5f * (1.0f + tanh(sum * 0.7978845608f * (1.0f + 0.044715f * sum * sum)));
    }
}

torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);

    auto output = torch::zeros({batch_size, output_size}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * output_size + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, input_size, output_size, divisor);

    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float divisor);"

fused_op = load_inline(
    name='fused_op',
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=['fused_operation_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation using a fused CUDA kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.divisor = divisor
        self.fused_op = fused_op

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        weight = self.linear.weight
        bias = self.linear.bias
        x = self.fused_op.fused_operation_cuda(x, weight, bias, self.divisor)
        return x