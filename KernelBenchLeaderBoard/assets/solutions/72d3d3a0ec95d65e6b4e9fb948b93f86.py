import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm, ReLU, and Conv2D
fused_bn_relu_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_relu_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels, int height, int width,
    int kernel_size, int padding, int stride, float eps) {

    // Calculate the index of the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * height * width;

    if (idx < total_elements) {
        // Calculate the position in the output tensor
        int w = idx % width;
        int h = (idx / width) % height;
        int c_out = (idx / (width * height)) % out_channels;
        int n = idx / (width * height * out_channels);

        // Initialize the output value
        float value = 0.0;

        // Perform the convolution
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

        // Apply batch normalization
        float mean = running_mean[c_out];
        float var = running_var[c_out];
        value = (value - mean) / sqrt(var + eps);

        // Apply ReLU
        value = fmaxf(0.0, value);

        // Add bias
        value += bias[c_out];

        // Store the result
        output[idx] = value;
    }
}

torch::Tensor fused_bn_relu_conv_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    int kernel_size, int padding, int stride, float eps) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;

    fused_bn_relu_conv_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width,
        kernel_size, padding, stride, eps);

    return output;
}
"""

fused_bn_relu_conv_cpp_source = """
torch::Tensor fused_bn_relu_conv_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    int kernel_size, int padding, int stride, float eps);
"""

# Compile the inline CUDA code for fused BatchNorm, ReLU, and Conv2D
fused_bn_relu_conv = load_inline(
    name='fused_bn_relu_conv',
    cpp_sources=fused_bn_relu_conv_cpp_source,
    cuda_sources=fused_bn_relu_conv_source,
    functions=['fused_bn_relu_conv_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockNew, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayerNew, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 48, 32]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayerNew(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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