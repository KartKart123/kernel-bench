import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel for softmax + bias + scale + sigmoid
__global__ void fused_softmax_bias_scale_sigmoid_kernel(
    float* output, const float* input, const float* bias,
    const int batch_size, const int channels, const int height, const int width,
    const float scale) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * height * width;
    
    if (idx < total_size) {
        const int w = idx % width;
        const int h = (idx / width) % height;
        const int c = (idx / (width * height)) % channels;
        const int b = idx / (width * height * channels);
        
        // Compute max for softmax
        float max_val = -INFINITY;
        for(int ch = 0; ch < channels; ch++) {
            const int offset = b * channels * height * width + ch * height * width + h * width + w;
            max_val = max(max_val, input[offset]);
        }
        
        // Compute sum for softmax
        float sum = 0.0f;
        for(int ch = 0; ch < channels; ch++) {
            const int offset = b * channels * height * width + ch * height * width + h * width + w;
            sum += expf(input[offset] - max_val);
        }
        
        // Compute final result with fused ops
        const int curr_idx = b * channels * height * width + c * height * width + h * width + w;
        float val = expf(input[curr_idx] - max_val) / sum;
        val = val + bias[c];
        val = val * scale;
        output[curr_idx] = 1.0f / (1.0f + expf(-val));
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scale) {
    const int batch_size = input.size(0);
    const int channels = input.size(1); 
    const int height = input.size(2);
    const int width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (batch_size * channels * height * width + threads - 1) / threads;
    
    fused_softmax_bias_scale_sigmoid_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size, channels, height, width,
        scale
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_ops_cuda(torch::Tensor input, torch::Tensor bias, float scale);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=padding, 
                                               output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_ops.fused_ops_cuda(x, self.bias.view(-1), self.scaling_factor)