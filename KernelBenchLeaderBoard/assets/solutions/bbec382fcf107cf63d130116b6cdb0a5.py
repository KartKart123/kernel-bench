import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for linear layer + ReLU
linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_relu_kernel(const float* input, const float* weight, const float* bias, float* output, int batch_size, int input_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * output_size) {
        int batch_index = i / output_size;
        int output_index = i % output_size;
        float sum = 0;
        for (int j = 0; j < input_size; ++j) {
            sum += input[batch_index * input_size + j] * weight[output_index * input_size + j];
        }
        sum += bias[output_index];
        output[i] = fmaxf(0.0f, sum); // ReLU activation
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int input_size = input.size(1);
    int output_size = weight.size(0);

    auto output = torch::zeros({batch_size, output_size}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * output_size + block_size - 1) / block_size;

    linear_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), batch_size, input_size, output_size);

    return output;
}
"""

linear_relu_cpp_source = "torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

linear_relu = load_inline(
    name='linear_relu',
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=['linear_relu_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ModelNew, self).__init__()
        self.linear_relu = linear_relu
        self.layers = nn.ModuleList()
        current_input_size = input_size
        for i, layer_size in enumerate(layer_sizes):
            self.layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size

        self.final_linear = nn.Linear(current_input_size, output_size)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.linear_relu.linear_relu_cuda(x, layer.weight, layer.bias)
        return self.final_linear(x)