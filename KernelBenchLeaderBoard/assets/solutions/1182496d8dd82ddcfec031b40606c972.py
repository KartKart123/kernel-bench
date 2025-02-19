import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Linear + ReLU
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_relu_forward(
    const float* __restrict__ input,        // [batch_size, in_features]
    const float* __restrict__ weight,       // [out_features, in_features]
    const float* __restrict__ bias,         // [out_features]
    float* __restrict__ output,             // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features) {

    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < batch_size && i < out_features) {
        float acc = bias[i];
        for (int j = 0; j < in_features; ++j) {
            acc += input[n * in_features + j] * weight[i * in_features + j];
        }
        // Apply ReLU activation
        output[n * out_features + i] = fmaxf(acc, 0.0f);
    }
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, out_features}, options);

    const dim3 blockSize(16, 16);
    const dim3 gridSize((out_features + blockSize.x - 1) / blockSize.x,
                         (batch_size + blockSize.y - 1) / blockSize.y);

    fused_linear_relu_forward<<<gridSize, blockSize>>>(
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

fused_linear_relu_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_linear_relu = load_inline(
    name='fused_linear_relu',
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=['fused_linear_relu_cuda'],
    verbose=True,
)

class FusedLinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = fused_linear_relu.fused_linear_relu_cuda(input, weight, bias)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is not implemented
        raise NotImplementedError("Backward pass not implemented.")

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
        # Following nn.Linear initialization
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return FusedLinearReLUFunction.apply(input, self.weight, self.bias)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Replacing fully connected layers with custom fused layers
        self.dropout1 = nn.Dropout(p=0.0)
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc1_fused = FusedLinearReLU(in_features=256 * 6 * 6, out_features=4096)
        self.fc2_fused = FusedLinearReLU(in_features=4096, out_features=4096)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1_fused(x)
        x = self.dropout1(x)
        
        x = self.fc2_fused(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x