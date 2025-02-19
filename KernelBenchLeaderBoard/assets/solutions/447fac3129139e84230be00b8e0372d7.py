import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused LeakyReLU + Add + Clamp + GELU kernel
__global__ void fused_ops_kernel(
    float* input,
    const float* sum_tensor,
    const int batch_size,
    const int channels,
    const int depth,
    const int height, 
    const int width) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * depth * height * width;
    
    if (idx < total_size) {
        const int c = (idx / (depth * height * width)) % channels;
        
        // LeakyReLU
        float val = input[idx];
        val = val > 0 ? val : 0.2f * val;
        
        // Add bias
        val += sum_tensor[c];
        
        // Clamp
        val = fminf(1.0f, fmaxf(-1.0f, val));
        
        // GELU approximation
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        float x3 = val * val * val;
        float inner = sqrt_2_over_pi * (val + coef * x3);
        val = 0.5f * val * (1.0f + tanhf(inner));
        
        input[idx] = val;
    }
}

void fused_ops_cuda(
    torch::Tensor input,
    const torch::Tensor sum_tensor) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1); 
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    const int total_elements = batch_size * channels * depth * height * width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_ops_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        batch_size,
        channels, 
        depth,
        height,
        width);
}
"""

cpp_source = """
void fused_ops_cuda(
    torch::Tensor input,
    const torch::Tensor sum_tensor);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        self.fused_ops.fused_ops_cuda(x, self.sum_tensor)
        return x