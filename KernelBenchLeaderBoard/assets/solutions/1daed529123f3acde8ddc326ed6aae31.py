import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Linear + ReLU
linear_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void linear_relu_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output feature index

    if (row < batch_size && col < out_features) {
        float value = bias[col];
        for (int i = 0; i < in_features; ++i) {
            value += input[row * in_features + i] * weight[col * in_features + i];
        }
        output[row * out_features + col] = fmaxf(value, 0.0f); // ReLU activation
    }
}

torch::Tensor linear_relu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias)
{
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (out_features + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linear_relu_forward_kernel<<<numBlocks, threadsPerBlock>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features);

    return output;
}
"""

linear_relu_cpp_source = """
torch::Tensor linear_relu_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);
"""

# Compile the inline CUDA code for Linear + ReLU
linear_relu = load_inline(
    name='linear_relu',
    cpp_sources=[linear_relu_cpp_source],
    cuda_sources=[linear_relu_source],
    functions=['linear_relu_forward_cuda'],
    verbose=True
)

# Define the custom LinearReLU module
import math

class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Define weight and bias as PyTorch parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # Initialize parameters
        self.reset_parameters()
        self.linear_relu = linear_relu

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Same initialization as nn.Linear
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if not input.is_cuda:
            input = input.cuda()
        output = self.linear_relu.linear_relu_forward_cuda(input, self.weight, self.bias)
        return output

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # VGG16 architecture with Conv2d and ReLU layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers using custom LinearReLU module
        self.classifier = nn.Sequential(
            LinearReLU(512 * 7 * 7, 4096),
            nn.Dropout(p=0.0),
            LinearReLU(4096, 4096),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x