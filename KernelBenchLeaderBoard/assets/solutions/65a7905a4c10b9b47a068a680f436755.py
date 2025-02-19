import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void custom_elementwise_kernel(const float* input, float* output, float subtract_value, float multiply_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        x = (x - subtract_value) * multiply_value;
        x = fmaxf(0.0f, x);  // ReLU activation
        output[idx] = x;
    }
}

torch::Tensor custom_elementwise_cuda(torch::Tensor input, float subtract_value, float multiply_value) {
    auto output = torch::zeros_like(input);
    int size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    custom_elementwise_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract_value,
        multiply_value,
        size
    );

    return output;
}
"""

cpp_source = """
torch::Tensor custom_elementwise_cuda(torch::Tensor input, float subtract_value, float multiply_value);
"""

custom_elementwise = load_inline(
    name='custom_elementwise',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['custom_elementwise_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.custom_elementwise = custom_elementwise

    def forward(self, x):
        x = self.linear(x)
        x = self.custom_elementwise.custom_elementwise_cuda(x, self.subtract_value, self.multiply_value)
        return x

batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]