import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for fused bias add + residual + multiply + residual
__global__ void fused_ops_kernel(float* x, const float* original_x, const float* bias,
                               int n, int c, int d, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = n * c * d * h * w;
    
    if (idx < total_size) {
        int c_idx = (idx / (d * h * w)) % c;
        float bias_val = bias[c_idx];
        float orig_val = original_x[idx];
        float x_val = x[idx];
        
        // Fused operations:
        // 1. Add bias
        // 2. Add residual
        // 3. Multiply with residual
        // 4. Add residual again
        x[idx] = ((x_val + bias_val + orig_val) * orig_val) + orig_val;
    }
}

std::vector<torch::Tensor> fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor bias) {
    
    auto original_x = x.clone();
    auto sizes = x.sizes();
    int n = sizes[0];
    int c = sizes[1];
    int d = sizes[2];
    int h = sizes[3];
    int w = sizes[4];
    
    const int threads = 256;
    const int blocks = (n * c * d * h * w + threads - 1) / threads;
    
    fused_ops_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        original_x.data_ptr<float>(),
        bias.data_ptr<float>(),
        n, c, d, h, w
    );
    
    return {x};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor bias);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=padding, 
                                               output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias)[0]
        return x