import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matmul + sigmoid + sum
matmul_sigmoid_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void matmul_sigmoid_sum_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int input_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float sum = 0;
        for (int j = 0; j < hidden_size; ++j) {
            float val = 0;
            for (int k = 0; k < input_size; ++k) {
                val += input[i * input_size + k] * weight[j * input_size + k];
            }
            val += bias[j];
            sum += 1.0f / (1.0f + expf(-val));
        }
        output[i] = sum;
    }
}

torch::Tensor matmul_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int hidden_size = weight.size(0);
    auto output = torch::zeros({batch_size, 1}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    matmul_sigmoid_sum_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, input_size, hidden_size);

    return output;
}
"""

matmul_sigmoid_sum_cpp_source = "torch::Tensor matmul_sigmoid_sum_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# Compile the inline CUDA code
matmul_sigmoid_sum = load_inline(
    name='matmul_sigmoid_sum',
    cpp_sources=matmul_sigmoid_sum_cpp_source,
    cuda_sources=matmul_sigmoid_sum_source,
    functions=['matmul_sigmoid_sum_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size, bias=True)
        self.matmul_sigmoid_sum = matmul_sigmoid_sum

    def forward(self, x):
        weight = self.linear.weight
        bias = self.linear.bias
        return self.matmul_sigmoid_sum.matmul_sigmoid_sum_cuda(x, weight, bias)