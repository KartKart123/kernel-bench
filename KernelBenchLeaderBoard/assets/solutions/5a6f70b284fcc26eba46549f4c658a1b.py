import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Conv2d + BatchNorm2d + ReLU
conv_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_bn_relu_kernel(const float* input, const float* weight, const float* bias, const float* bn_weight, const float* bn_bias, const float* bn_running_mean, const float* bn_running_var, float* output, int in_channels, int out_channels, int height, int width, int kernel_size, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < out_channels; ++c) {
            float sum = 0;
            for (int k = 0; k < in_channels; ++k) {
                for (int i = -padding; i < kernel_size - padding; ++i) {
                    for (int j = -padding; j < kernel_size - padding; ++j) {
                        int input_x = x + j;
                        int input_y = y + i;
                        if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
                            sum += input[k * height * width + input_y * width + input_x] * weight[c * in_channels * kernel_size * kernel_size + k * kernel_size * kernel_size + (i + padding) * kernel_size + (j + padding)];
                        }
                    }
                }
            }
            float bn_input = (sum + bias[c]) ;
            float bn_output = bn_weight[c] * (bn_input - bn_running_mean[c]) / sqrtf(bn_running_var[c] + 1e-5) + bn_bias[c];
            output[c * height * width + y * width + x] = fmaxf(0.0f, bn_output);
        }
    }
}


torch::Tensor conv_bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_running_mean, torch::Tensor bn_running_var) {
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int kernel_size = weight.size(2);
    int padding = kernel_size / 2;

    auto output = torch::zeros({input.size(0), out_channels, height, width}, input.options());

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    conv_bn_relu_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(), bn_running_mean.data_ptr<float>(), bn_running_var.data_ptr<float>(), output.data_ptr<float>(), in_channels, out_channels, height, width, kernel_size, padding);

    return output;
}
"""

conv_bn_relu_cpp_source = "torch::Tensor conv_bn_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_running_mean, torch::Tensor bn_running_var);"

conv_bn_relu = load_inline(
    name='conv_bn_relu',
    cpp_sources=conv_bn_relu_cpp_source,
    cuda_sources=conv_bn_relu_source,
    functions=['conv_bn_relu_cuda'],
    verbose=True
)


# Custom CUDA kernel for Global Average Pooling
global_avg_pool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool_kernel(const float* input, float* output, int batch_size, int channels, int height, int width) {
    int b = blockIdx.x;
    int c = threadIdx.x;

    if (b < batch_size && c < channels) {
        float sum = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                sum += input[b * channels * height * width + c * height * width + i * width + j];
            }
        }
        output[b * channels + c] = sum / (height * width);
    }
}

torch::Tensor global_avg_pool_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::zeros({batch_size, channels}, input.options());

    dim3 block_size(channels);
    dim3 grid_size(batch_size);

    global_avg_pool_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}
"""

global_avg_pool_cpp_source = "torch::Tensor global_avg_pool_cuda(torch::Tensor input);"

global_avg_pool = load_inline(
    name='global_avg_pool',
    cpp_sources=global_avg_pool_cpp_source,
    cuda_sources=global_avg_pool_source,
    functions=['global_avg_pool_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(ModelNew, self).__init__()
        self.stages = stages
        self.block_widths = block_widths
        self.conv_bn_relu = conv_bn_relu
        self.global_avg_pool = global_avg_pool

        layers = []
        current_channels = input_channels
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(block_widths[-1], output_classes)

    def _make_stage(self, in_channels, out_channels):
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
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Conv2d) and isinstance(layer[1], nn.BatchNorm2d) and isinstance(layer[2], nn.ReLU):
                x = self.conv_bn_relu.conv_bn_relu_cuda(x, layer[0].weight, layer[0].bias, layer[1].weight, layer[1].bias, layer[1].running_mean, layer[1].running_var)
            else:
                x = layer(x)
        x = self.global_avg_pool.global_avg_pool_cuda(x)
        x = self.fc(x)
        return x