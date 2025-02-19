import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Linear + ReLU
linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_relu_kernel(const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ y, int batch_size, int in_features, int out_features) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && out_idx < out_features) {
        float val = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            val += x[batch_idx * in_features + k] * weight[out_idx * in_features + k];
        }
        val += bias[out_idx];
        y[batch_idx * out_features + out_idx] = val > 0.0f ? val : 0.0f; // ReLU activation
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto y = torch::empty({batch_size, out_features}, x.options());

    dim3 blockDim(16, 16);
    dim3 gridDim((out_features + blockDim.x - 1) / blockDim.x, (batch_size + blockDim.y - 1) / blockDim.y);

    linear_relu_kernel<<<gridDim, blockDim>>>(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), batch_size, in_features, out_features);

    return y;
}
"""

linear_relu_cpp_source = """
torch::Tensor linear_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code for Linear + ReLU
linear_relu = load_inline(
    name='linear_relu',
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=['linear_relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Register parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return linear_relu.linear_relu_cuda(x, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(LinearReLU(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        # Last layer without ReLU
        self.final_linear = nn.Linear(current_input_size, output_size)
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        for layer in self.layers:
            x = layer(x)
        x = self.final_linear(x)
        return x