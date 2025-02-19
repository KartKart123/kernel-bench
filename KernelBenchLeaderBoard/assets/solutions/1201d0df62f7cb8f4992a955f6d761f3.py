import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activation functions
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_activation_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Swish: x = x * sigmoid(x)
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        x = x * sigmoid_x;
        // Tanh: x = tanh(x)
        x = tanhf(x);
        // GELU: x = x * 0.5f * (1.0f + erf(x / sqrt(2)))
        x = x * 0.5f * (1.0f + erff(x * 0.70710678f)); // 1/sqrt(2) ≈ 0.70710678
        // Hardtanh: x = clamp(x, -1.0f, 1.0f)
        x = fminf(fmaxf(x, -1.0f), 1.0f);
        output[idx] = x;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input) {
    int size = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    fused_activation_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_activation_cpp_source = "torch::Tensor fused_activation_cuda(torch::Tensor input);"

# Compile the inline CUDA code for fused activation functions
fused_activation = load_inline(
    name='fused_activation',
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=['fused_activation_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O2'],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom fused activation CUDA kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = self.fused_activation.fused_activation_cuda(x)
        return x

batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]