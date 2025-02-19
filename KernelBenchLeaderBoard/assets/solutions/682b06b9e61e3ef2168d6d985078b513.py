import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + ReLU
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_relu_forward_kernel(const float* __restrict__ x,
                                          const float* __restrict__ weight,
                                          const float* __restrict__ bias,
                                          float* __restrict__ out,
                                          int batch_size,
                                          int input_size,
                                          int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * output_size;

    if (idx < total) {
        int batch_idx = idx / output_size;
        int out_idx = idx % output_size;

        float sum = bias[out_idx];
        for (int k = 0; k < input_size; ++k) {
            sum += x[batch_idx * input_size + k] * weight[out_idx * input_size + k];
        }

        // Apply ReLU
        out[batch_idx * output_size + out_idx] = sum > 0 ? sum : 0;
    }
}

torch::Tensor linear_relu_forward_cuda(torch::Tensor x,
                                       torch::Tensor weight,
                                       torch::Tensor bias) {
    int batch_size = x.size(0);
    int input_size = x.size(1);
    int output_size = weight.size(0);

    auto out = torch::empty({batch_size, output_size}, x.options());

    int threads = 256;
    int blocks = (batch_size * output_size + threads - 1) / threads;

    linear_relu_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        output_size);

    return out;
}
"""

cpp_source = """
torch::Tensor linear_relu_forward_cuda(torch::Tensor x,
                                       torch::Tensor weight,
                                       torch::Tensor bias);
"""

# Compile the inline CUDA code
linear_relu = load_inline(
    name='linear_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['linear_relu_forward_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
)

import math

class FusedLinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super(FusedLinearReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return linear_relu.linear_relu_forward_cuda(x, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Feature layers (same as original Model)
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
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Replace Linear + ReLU layers with FusedLinearReLU
        self.classifier = nn.Sequential(
            FusedLinearReLU(512 * 7 * 7, 4096),
            nn.Dropout(p=0.0),
            FusedLinearReLU(4096, 4096),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x