import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for fused mean + bias + softmax + tanh + scale
__global__ void fused_ops_kernel(
    const float* input,
    const float* bias,
    float* output,
    const int batch_size,
    const int channels,
    const int depth,
    const int height, 
    const int width,
    const float scale_factor) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * depth * height * width;
    
    if (idx < total_elements) {
        // Calculate positions
        const int w = idx % width;
        const int h = (idx / width) % height;
        const int d = (idx / (width * height)) % depth;
        const int b = idx / (width * height * depth);
        
        // Calculate mean across channels
        float sum = 0.0f;
        for(int c = 0; c < channels; c++) {
            sum += input[((b * channels + c) * depth + d) * height * width + h * width + w];
        }
        float mean = sum / channels;
        
        // Add bias
        float val = mean + bias[0];
        
        // Apply softmax
        float exp_val = expf(val);
        float sum_exp = exp_val;
        float softmax = exp_val / sum_exp;
        
        // Apply tanh and scale
        output[idx] = tanhf(softmax) * scale_factor;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scale_factor) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    auto output = torch::zeros({batch_size, 1, depth, height, width}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * depth * height * width + threads - 1) / threads;
    
    fused_ops_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels, 
        depth,
        height,
        width,
        scale_factor
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor bias, 
    float scale_factor);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_ops.fused_ops_cuda(x, self.bias, self.scaling_factor)