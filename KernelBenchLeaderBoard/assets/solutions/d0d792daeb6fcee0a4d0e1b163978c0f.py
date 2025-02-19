import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_impl(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__device__ float sigmoid_impl(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_activations_bias_kernel(
    float* input,
    const float* bias,
    const int n_elements,
    const int channels,
    const int dhw) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int c = (idx / dhw) % channels;
        float val = input[idx];
        
        // ReLU
        val = fmaxf(val, 0.0f);
        
        // LeakyReLU
        val = val > 0 ? val : 0.01f * val;
        
        // GELU
        val = gelu_impl(val);
        
        // Sigmoid
        val = sigmoid_impl(val);
        
        // Add bias
        val += bias[c];
        
        input[idx] = val;
    }
}

void fused_activations_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias) {
    
    const int n_elements = input.numel();
    const int channels = input.size(1);
    const int dhw = input.size(2) * input.size(3) * input.size(4);
    
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;
    
    fused_activations_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        n_elements,
        channels, 
        dhw);
}
"""

cpp_source = """
void fused_activations_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_activations_bias_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        self.fused_ops.fused_activations_bias_cuda(x, self.bias)
        return x