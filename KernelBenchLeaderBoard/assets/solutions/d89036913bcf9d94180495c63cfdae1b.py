import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA code for fused BatchNorm2d and ReLU
fused_bn_relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void fused_bn_relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int C, int HW) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = C * HW;
    if (idx >= N) return;
    int c = (idx / HW) % C;

    float mean = running_mean[c];
    float var = running_var[c];
    float gamma = weight[c];
    float beta = bias[c];

    float inv_std = rsqrtf(var + eps);
    float val = (x[idx] - mean) * inv_std * gamma + beta;
    y[idx] = val > 0 ? val : 0;  // ReLU
}

torch::Tensor fused_bn_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int HW = H * W;
    int numel = N * C * HW;

    auto y = torch::empty_like(x);

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    fused_bn_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        C, HW);

    return y;
}
"""

fused_bn_relu_cpp_source = """
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);
"""

# Compile the fused batchnorm ReLU CUDA code
fused_bn_relu = load_inline(
    name='fused_bn_relu',
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_cuda_source,
    functions=['fused_bn_relu_cuda'],
    verbose=False,
)

class FusedBatchNormReLU2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FusedBatchNormReLU2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_parameter('weight', nn.Parameter(torch.ones(num_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_features)))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Use the standard BatchNorm during training
            x = F.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias,
                training=True, momentum=0.1, eps=self.eps)
            x = F.relu(x, inplace=True)
            return x
        else:
            # Use the fused operation during inference
            return fused_bn_relu.fused_bn_relu_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(self._make_layer(in_features, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with FusedBatchNormReLU2d, Conv2D, and Dropout.
        """
        return nn.Sequential(
            FusedBatchNormReLU2d(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)  # Concatenate along channel axis
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            FusedBatchNormReLU2d(num_input_features),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(x)

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        """
        :param growth_rate: The growth rate of the DenseNet (new features added per layer)
        :param num_classes: The number of output classes for classification
        """
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 24, 16]  # Corresponding layers in DenseNet121

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = FusedBatchNormReLU2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)  # Already included in FusedBatchNormReLU2d
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x