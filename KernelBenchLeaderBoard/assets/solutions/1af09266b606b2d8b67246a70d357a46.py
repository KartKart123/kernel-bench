import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused depthwise convolution + batch norm + ReLU
fused_conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_bn_relu_kernel(
    const float* input, const float* weight, const float* bias, const float* running_mean, const float* running_var, 
    float* output, int batch_size, int channels, int height, int width, int kernel_size, int stride, int padding, 
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        // Calculate the position in the output tensor
        int w_out = idx % width;
        int h_out = (idx / width) % height;
        int c_out = (idx / (width * height)) % channels;
        int b = idx / (width * height * channels);
        
        // Initialize the output value
        float value = 0.0;
        
        // Perform depthwise convolution
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((b * channels + c_out) * height + h_in) * width + w_in;
                    int weight_idx = (c_out * kernel_size + kh) * kernel_size + kw;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        // Apply batch normalization
        value = (value - running_mean[c_out]) / sqrt(running_var[c_out] + eps);
        
        // Add bias
        value += bias[c_out];
        
        // Apply ReLU
        value = fmaxf(value, 0.0f);
        
        // Store the result
        output[idx] = value;
    }
}

torch::Tensor fused_conv_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, 
    int kernel_size, int stride, int padding, float eps) {
    
    auto output = torch::zeros_like(input);
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    const int block_size = 256;
    const int num_blocks = (batch_size * channels * height * width + block_size - 1) / block_size;
    
    fused_conv_bn_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(), 
        output.data_ptr<float>(), batch_size, channels, height, width, 
        kernel_size, stride, padding, eps);
    
    return output;
}
"""

fused_conv_bn_relu_cpp_source = "torch::Tensor fused_conv_bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, int kernel_size, int stride, int padding, float eps);"

# Compile the inline CUDA code for fused operations
fused_conv_bn_relu = load_inline(
    name='fused_conv_bn_relu',
    cpp_sources=fused_conv_bn_relu_cpp_source,
    cuda_sources=fused_conv_bn_relu_source,
    functions=['fused_conv_bn_relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()
        
        def conv_bn_fused(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw_fused(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn_fused(input_channels, int(32 * alpha), 2),
            conv_dw_fused(int(32 * alpha), int(64 * alpha), 1),
            conv_dw_fused(int(64 * alpha), int(128 * alpha), 2),
            conv_dw_fused(int(128 * alpha), int(128 * alpha), 1),
            conv_dw_fused(int(128 * alpha), int(256 * alpha), 2),
            conv_dw_fused(int(256 * alpha), int(256 * alpha), 1),
            conv_dw_fused(int(256 * alpha), int(512 * alpha), 2),
            conv_dw_fused(int(512 * alpha), int(512 * alpha), 1),
            conv_dw_fused(int(512 * alpha), int(512 * alpha), 1),
            conv_dw_fused(int(512 * alpha), int(512 * alpha), 1),
            conv_dw_fused(int(512 * alpha), int(512 * alpha), 1),
            conv_dw_fused(int(512 * alpha), int(512 * alpha), 1),
            conv_dw_fused(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw_fused(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x