import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU
fused_conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_relu_kernel(const float* input, const float* weight, const float* bias, float* output, 
                                       int batch_size, int in_channels, int out_channels, int height, int width, 
                                       int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    int output_size = batch_size * out_channels * output_height * output_width;

    if (idx < output_size) {
        int w_out = idx % output_width;
        int h_out = (idx / output_width) % output_height;
        int c_out = (idx / (output_width * output_height)) % out_channels;
        int n = idx / (output_width * output_height * out_channels);

        float value = 0.0;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = n * in_channels * height * width + c_in * height * width + h_in * width + w_in;
                        int weight_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        value += bias[c_out];
        output[idx] = fmaxf(value, 0.0); // ReLU
    }
}

torch::Tensor fused_conv_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                   int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);

    int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * output_height * output_width + block_size - 1) / block_size;

    fused_conv_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), 
        batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding
    );

    return output;
}
"""

fused_conv_relu_cpp_source = "torch::Tensor fused_conv_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"

# Compile the inline CUDA code for fused Conv2d + ReLU
fused_conv_relu = load_inline(
    name='fused_conv_relu',
    cpp_sources=fused_conv_relu_cpp_source,
    cuda_sources=fused_conv_relu_source,
    functions=['fused_conv_relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class FireModuleNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModuleNew, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.squeeze.weight, self.squeeze.bias, 1, 0)
        return torch.cat([
            fused_conv_relu.fused_conv_relu_cuda(x, self.expand1x1.weight, self.expand1x1.bias, 1, 0),
            fused_conv_relu.fused_conv_relu_cuda(x, self.expand3x3.weight, self.expand3x3.bias, 1, 1)
        ], 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(96, 16, 64, 64),
            FireModuleNew(128, 16, 64, 64),
            FireModuleNew(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(256, 32, 128, 128),
            FireModuleNew(256, 48, 192, 192),
            FireModuleNew(384, 48, 192, 192),
            FireModuleNew(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModuleNew(512, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)