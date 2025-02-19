import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Conv2d + ReLU
fused_conv_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_relu_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int padding, int stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;

    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c_out = (idx / (width * height)) % out_channels;
        int n = idx / (width * height * out_channels);

        float value = 0.0;
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = h * stride - padding + kh;
                    int w_in = w * stride - padding + kw;
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = ((n * in_channels + c_in) * height + h_in) * width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        value += bias[c_out];
        output[idx] = fmaxf(value, 0.0); // ReLU
    }
}

torch::Tensor fused_conv_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int padding, int stride) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;

    fused_conv_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width,
        kernel_size, padding, stride);

    return output;
}
"""

fused_conv_relu_cpp_source = "torch::Tensor fused_conv_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_size, int padding, int stride);"

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

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # VGG16 architecture: 5 blocks of convolutional layers followed by max pooling
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv1_1.weight, self.conv1_1.bias, 3, 1, 1)
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv1_2.weight, self.conv1_2.bias, 3, 1, 1)
        x = self.maxpool1(x)
        
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv2_1.weight, self.conv2_1.bias, 3, 1, 1)
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv2_2.weight, self.conv2_2.bias, 3, 1, 1)
        x = self.maxpool2(x)
        
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv3_1.weight, self.conv3_1.bias, 3, 1, 1)
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv3_2.weight, self.conv3_2.bias, 3, 1, 1)
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv3_3.weight, self.conv3_3.bias, 3, 1, 1)
        x = self.maxpool3(x)
        
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv4_1.weight, self.conv4_1.bias, 3, 1, 1)
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv4_2.weight, self.conv4_2.bias, 3, 1, 1)
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv4_3.weight, self.conv4_3.bias, 3, 1, 1)
        x = self.maxpool4(x)
        
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv5_1.weight, self.conv5_1.bias, 3, 1, 1)
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv5_2.weight, self.conv5_2.bias, 3, 1, 1)
        x = fused_conv_relu.fused_conv_relu_cuda(x, self.conv5_3.weight, self.conv5_3.bias, 3, 1, 1)
        x = self.maxpool5(x)
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x