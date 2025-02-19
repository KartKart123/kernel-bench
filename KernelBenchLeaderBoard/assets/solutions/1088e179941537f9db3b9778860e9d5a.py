import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU6
relu6_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu6_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = fminf(fmaxf(x, 0.0f), 6.0f);
    }
}

torch::Tensor relu6_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);

    int size = input.numel();
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    relu6_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

relu6_cpp_source = "torch::Tensor relu6_cuda(torch::Tensor input);"

# Compile the inline CUDA code for ReLU6
relu6 = load_inline(
    name='relu6_cuda',
    cpp_sources=relu6_cpp_source,
    cuda_sources=relu6_source,
    functions=['relu6_cuda'],
    verbose=True,
)

# Define a custom Module that uses the custom ReLU6
class CustomReLU6(nn.Module):
    def __init__(self):
        super(CustomReLU6, self).__init__()
        self.relu6_func = relu6.relu6_cuda

    def forward(self, input):
        return self.relu6_func(input)

# Now define ModelNew with replacements
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        MobileNetV2 architecture implementation in PyTorch with custom ReLU6 operator.
        """
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            """
            Inverted Residual Block for MobileNetV2.
            """
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(CustomReLU6())

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                CustomReLU6(),
                # Pointwise linear convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    CustomReLU6()]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride_i = s if i == 0 else 1
                block, use_res_connect = _inverted_residual_block(input_channel, output_channel, stride_i, expand_ratio=t)
                features.append(block)
                input_channel = output_channel

        # Building last several layers
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(CustomReLU6())

        # Final layer
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
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
        """
        Forward pass of the MobileNetV2 model with custom ReLU6.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x