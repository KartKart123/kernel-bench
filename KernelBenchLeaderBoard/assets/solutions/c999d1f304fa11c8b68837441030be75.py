import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused MaxPool + Hardtanh kernel
__global__ void maxpool_hardtanh_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels, 
    const int height,
    const int width,
    const int pool_size,
    const int pool_stride,
    const float min_val,
    const float max_val,
    const int out_height,
    const int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * channels * out_height * out_width) return;
    
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);
    
    float maxval = -1e10;
    
    int h_start = h_out * pool_stride;
    int w_start = w_out * pool_stride;
    
    for(int i = 0; i < pool_size; i++) {
        for(int j = 0; j < pool_size; j++) {
            int h_pos = h_start + i;
            int w_pos = w_start + j;
            if(h_pos < height && w_pos < width) {
                float val = input[((b * channels + c) * height + h_pos) * width + w_pos];
                maxval = max(maxval, val);
            }
        }
    }
    
    // Apply hardtanh
    maxval = min(max(maxval, min_val), max_val);
    
    output[idx] = maxval;
}

// Fused Mean + Tanh kernel
__global__ void mean_tanh_kernel(
    const float* input,
    float* output, 
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size * channels) return;
    
    int c = idx % channels;
    int b = idx / channels;
    
    float sum = 0.0f;
    int count = height * width;
    
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            sum += input[((b * channels + c) * height + h) * width + w];
        }
    }
    
    float mean = sum / count;
    output[idx] = tanhf(mean);
}

torch::Tensor maxpool_hardtanh_cuda(
    torch::Tensor input,
    int pool_size,
    int pool_stride, 
    float min_val,
    float max_val
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int out_height = (height - pool_size) / pool_stride + 1;
    const int out_width = (width - pool_size) / pool_stride + 1;
    
    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * channels * out_height * out_width + threads - 1) / threads;
    
    maxpool_hardtanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        pool_size,
        pool_stride,
        min_val,
        max_val,
        out_height,
        out_width
    );
    
    return output;
}

torch::Tensor mean_tanh_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    auto output = torch::empty({batch_size, channels, 1, 1}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * channels + threads - 1) / threads;
    
    mean_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor maxpool_hardtanh_cuda(torch::Tensor input, int pool_size, int pool_stride, float min_val, float max_val);
torch::Tensor mean_tanh_cuda(torch::Tensor input);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['maxpool_hardtanh_cuda', 'mean_tanh_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.maxpool_hardtanh_cuda(x, self.maxpool_kernel_size, self.maxpool_stride, self.hardtanh_min, self.hardtanh_max)
        x = self.fused_ops.mean_tanh_cuda(x)
        return x