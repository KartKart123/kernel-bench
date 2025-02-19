import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for pointwise convolution
pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void pointwise_conv_kernel(const float* input, const float* weight, float* output, int in_channels, int out_channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = out_channels * height * width;

    if (idx < total_elements) {
        int oc = idx / (height * width);
        int hw = idx % (height * width);
        int h = hw / width;
        int w = hw % width;

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            sum += input[ic * height * width + h * width + w] * weight[oc * in_channels + ic];
        }
        output[idx] = sum;
    }
}

torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight) {
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros({input.size(0), out_channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (out_channels * height * width + block_size - 1) / block_size;

    pointwise_conv_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), in_channels, out_channels, height, width);

    return output;
}
"""

pointwise_conv_cpp_source = "torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight);"

# Compile the inline CUDA code for pointwise convolution
pointwise_conv = load_inline(
    name='pointwise_conv',
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=['pointwise_conv_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv_kernel(const float* input, const float* weight, float* output, int channels, int height, int width, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = channels * height * width;

    if (idx < total_elements) {
        int c = idx / (height * width);
        int hw = idx % (height * width);
        int h = hw / width;
        int w = hw % width;

        float sum = 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h * stride + kh - 1;
                int iw = w * stride + kw - 1;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    sum += input[c * height * width + ih * width + iw] * weight[c * 9 + kh * 3 + kw];
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor depthwise_conv_cuda(torch::Tensor input, torch::Tensor weight, int stride) {
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros({input.size(0), channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (channels * height * width + block_size - 1) / block_size;

    depthwise_conv_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), channels, height, width, stride);

    return output;
}
"""

depthwise_conv_cpp_source = "torch::Tensor depthwise_conv_cuda(torch::Tensor input, torch::Tensor weight, int stride);"

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
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)[0])
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x