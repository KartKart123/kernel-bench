import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
relu_source = """
#include <torch/extension.h>

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0 ? val : 0;
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int64_t size = input.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

relu_cpp_source = """
torch::Tensor relu_cuda(torch::Tensor input);
"""

# Compile the custom ReLU operator
relu_op = load_inline(
    name='relu_op',
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=['relu_cuda'],
    verbose=False
)

class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
    
    def forward(self, x):
        return relu_op.relu_cuda(x)

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(FireModule, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = CustomReLU()
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = CustomReLU()
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = CustomReLU()
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            CustomReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            CustomReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

def get_inputs():
    batch_size = 1
    input_channels = 3
    height = 224
    width = 224
    return [torch.randn(batch_size, input_channels, height, width).cuda()]

def get_init_inputs():
    num_classes = 1000
    return [num_classes]