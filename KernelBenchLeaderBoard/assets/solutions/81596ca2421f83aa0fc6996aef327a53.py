import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Global Average Pooling
global_avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool2d_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int batch_size,
                                         int channels,
                                         int height,
                                         int width) {
    int b = blockIdx.x;
    int c = threadIdx.x;

    if (b < batch_size && c < channels) {
        int idx = b * channels * height * width + c * height * width;
        float sum = 0.0f;

        for (int i = 0; i < height * width; ++i) {
            sum += input[idx + i];
        }

        output[b * channels + c] = sum / (height * width);
    }
}

torch::Tensor global_avg_pool2d_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);

    auto output = torch::zeros({batch_size, channels}, input.options());

    const dim3 blocks(batch_size);
    const dim3 threads(channels);

    global_avg_pool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return output;
}
"""

global_avg_pool2d_cpp_source = "torch::Tensor global_avg_pool2d_cuda(torch::Tensor input);"

# Compile the inline CUDA code for Global Average Pooling
global_avg_pool2d = load_inline(
    name='global_avg_pool2d',
    cpp_sources=global_avg_pool2d_cpp_source,
    cuda_sources=global_avg_pool2d_source,
    functions=['global_avg_pool2d_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
)

class GlobalAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = global_avg_pool2d.global_avg_pool2d_cuda(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        batch_size, channels, height, width = input.size()
        grad_input = grad_output.view(batch_size, channels, 1, 1).expand(batch_size, channels, height, width)
        grad_input = grad_input / (height * width)
        return grad_input

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, input):
        return GlobalAvgPool2dFunction.apply(input)

class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths

        layers = []
        current_channels = input_channels

        # Construct the stages with their respective blocks
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]

        self.feature_extractor = nn.Sequential(*layers)

        # Custom Global Average Pooling Layer
        self.global_avg_pool = GlobalAvgPool2d()

        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)

    def _make_stage(self, in_channels, out_channels):
        """
        Creates a simple block for each stage.
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return: nn.Sequential block with convolutional layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass through the optimized model.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        x = self.feature_extractor(x)
        x = self.global_avg_pool(x)  # Use custom CUDA Global Average Pooling
        x = self.fc(x)
        return x