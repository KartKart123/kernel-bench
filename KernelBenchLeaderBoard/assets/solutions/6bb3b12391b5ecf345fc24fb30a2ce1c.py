import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for concatenation along the channel axis
concat_channel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void concat_channel_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out,
                                      int batch_size,
                                      int channels_a,
                                      int channels_b,
                                      int height,
                                      int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * (channels_a + channels_b) * height * width;
    if (idx < total_elements) {
        int n = idx / ( (channels_a + channels_b) * height * width );
        int c = (idx / (height * width)) % (channels_a + channels_b);
        int h = (idx / width) % height;
        int w = idx % width;

        if (c < channels_a) {
            int a_idx = n * channels_a * height * width + c * height * width + h * width + w;
            out[idx] = a[a_idx];
        } else {
            int c_b = c - channels_a;
            int b_idx = n * channels_b * height * width + c_b * height * width + h * width + w;
            out[idx] = b[b_idx];
        }
    }
}

torch::Tensor concat_channel_cuda(torch::Tensor a, torch::Tensor b) {
    int batch_size = a.size(0);
    int channels_a = a.size(1);
    int channels_b = b.size(1);
    int height = a.size(2);
    int width = a.size(3);

    auto out = torch::empty({batch_size, channels_a + channels_b, height, width}, a.options());

    int total_elements = out.numel();

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    concat_channel_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
                                               batch_size, channels_a, channels_b, height, width);
    return out;
}
"""

concat_channel_cpp_source = "torch::Tensor concat_channel_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code for concatenation
concat_channel = load_inline(
    name='concat_channel',
    cpp_sources=concat_channel_cpp_source,
    cuda_sources=concat_channel_source,
    functions=['concat_channel_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)

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
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        for layer in self.layers:
            new_feature = layer(x)
            x = concat_channel.concat_channel_cuda(x, new_feature)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
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
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 48, 32]  # Corresponding layers in DenseNet201

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
        self.final_bn = nn.BatchNorm2d(num_features)
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
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x