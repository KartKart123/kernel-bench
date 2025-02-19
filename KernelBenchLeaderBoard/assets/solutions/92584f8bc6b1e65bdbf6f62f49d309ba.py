import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU
conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

__global__ void conv_relu_kernel(
    const float* input, float* output, const float* weight, const float* bias,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int stride, int padding, int dilation) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    for (int i = idx; i < batch_size * out_channels * out_height * out_width; i += num_threads) {
        int b = i / (out_channels * out_height * out_width);
        int c = (i / (out_height * out_width)) % out_channels;
        int h = (i / out_width) % out_height;
        int w = i % out_width;
        
        float sum = 0.0f;
        for (int k = 0; k < in_channels; ++k) {
            for (int p = 0; p < kernel_size; ++p) {
                for (int q = 0; q < kernel_size; ++q) {
                    int in_h = h * stride - padding + p * dilation;
                    int in_w = w * stride - padding + q * dilation;
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        sum += input[b * in_channels * height * width + k * height * width + in_h * width + in_w] *
                               weight[c * in_channels * kernel_size * kernel_size + k * kernel_size * kernel_size + p * kernel_size + q];
                    }
                }
            }
        }
        sum += bias[c];
        output[i] = max(sum, 0.0f);
    }
}

torch::Tensor conv_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int dilation) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto output = torch::zeros({batch_size, out_channels, (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1,
                                (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * output.size(2) * output.size(3) + block_size - 1) / block_size;
    
    conv_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, dilation);
    
    return output;
}
"""

conv_relu_cpp_source = "torch::Tensor conv_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation);"

# Compile the inline CUDA code for fused Conv2d + ReLU
conv_relu = load_inline(
    name='conv_relu',
    cpp_sources=conv_relu_cpp_source,
    cuda_sources=conv_relu_source,
    functions=['conv_relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.dropout1 = nn.Dropout(p=0.0)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        
        self.conv_relu = conv_relu
    
    def forward(self, x):
        x = self.conv_relu.conv_relu_cuda(x, self.conv1.weight, self.conv1.bias, 4, 2, 1)
        x = self.maxpool1(x)
        
        x = self.conv_relu.conv_relu_cuda(x, self.conv2.weight, self.conv2.bias, 1, 2, 1)
        x = self.maxpool2(x)
        
        x = self.conv_relu.conv_relu_cuda(x, self.conv3.weight, self.conv3.bias, 1, 1, 1)
        
        x = self.conv_relu.conv_relu_cuda(x, self.conv4.weight, self.conv4.bias, 1, 1, 1)
        
        x = self.conv_relu.conv_relu_cuda(x, self.conv5.weight, self.conv5.bias, 1, 1, 1)
        x = self.maxpool3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x