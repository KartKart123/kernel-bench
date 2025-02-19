import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU activation function
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
        float c = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        float tanh_c = tanhf(c);
        output[idx] = x * 0.5f * (1.0f + tanh_c);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor input);"

# Compile the inline CUDA code for GELU activation function
gelu_module = load_inline(
    name='gelu_module',
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=['gelu_cuda'],
    verbose=False,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gelu = gelu_module

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu.gelu_cuda(x)
        x = F.softmax(x, dim=1)
        return x

batch_size = 128
in_features = 100
out_features = 10

def get_inputs():
    return [torch.randn(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]