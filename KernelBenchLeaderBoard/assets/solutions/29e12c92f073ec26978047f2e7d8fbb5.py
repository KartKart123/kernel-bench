import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv_kernel(const float* input, const float* weight, float* output, 
                                      int in_channels, int out_channels, int height, int width, 
                                      int kernel_size, int stride, int padding) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    if (out_x < width && out_y < height) {
        float sum = 0.0f;
        for (int kx = 0; kx < kernel_size; ++kx) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                int in_x = out_x * stride + kx - padding;
                int in_y = out_y * stride + ky - padding;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    sum += input[(out_c * height + in_y) * width + in_x] * weight[out_c * kernel_size * kernel_size + kx * kernel_size + ky];
                }
            }
        }
        output[(out_c * height + out_y) * width + out_x] = sum;
    }
}

torch::Tensor depthwise_conv_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({input.size(0), out_channels, height, width}, input.options());

    const int block_size = 16;
    dim3 num_blocks((width + block_size - 1) / block_size, (height + block_size - 1) / block_size, out_channels);
    dim3 threads_per_block(block_size, block_size, 1);

    depthwise_conv_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        in_channels, out_channels, height, width, kernel_size, stride, padding
    );

    return output;
}
"""

depthwise_conv_cpp_source = "torch::Tensor depthwise_conv_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);"

# Compile the inline CUDA code for depthwise convolution
depthwise_conv = load_inline(
    name='depthwise_conv',
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=['depthwise_conv_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x