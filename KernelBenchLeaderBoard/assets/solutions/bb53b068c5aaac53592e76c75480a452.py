import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU
relu_source = """
#include <torch/extension.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = (input[i] > 0) ? input[i] : 0;
  }
}

torch::Tensor relu_cuda(torch::Tensor input) {
  auto size = input.numel();
  auto output = torch::zeros_like(input);
  const int threads_per_block = 256;
  const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
  relu_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
  return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

relu = load_inline(
    name='relu',
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=['relu_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)


# Define the custom CUDA kernel for group convolution
group_conv_source = """
#include <torch/extension.h>

__global__ void group_conv_kernel(const float* input, const float* weight, float* output, int N, int C, int H, int W, int K, int G, int P) {
    int n = blockIdx.x / (gridDim.y * gridDim.z);
    int g = (blockIdx.x % (gridDim.y * gridDim.z)) / gridDim.z;
    int k = (blockIdx.x % gridDim.z);
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= H + 2 * P || j >= W + 2 * P) return;

    int input_index = n * C * (H + 2 * P) * (W + 2 * P) + g * (C / G) * (H + 2 * P) * (W + 2 * P) + (i * (W + 2 * P) + j);
    int output_index = n * (C / G) * H * W + g * (C / G) * H * W + k * H * W + i * W + j;

    float sum = 0;
    for (int x = 0; x < K; ++x) {
        for (int y = 0; y < K; ++y) {
            int input_x = i + x - P;
            int input_y = j + y - P;
            if (input_x >= 0 && input_x < H && input_y >= 0 && input_y < W) {
                int weight_index = g * (C / G) * K * K + k * K * K + x * K + y;
                sum += input[input_index + x * (W + 2 * P) + y] * weight[weight_index];
            }
        }
    }
    output[output_index] = sum;
}

torch::Tensor group_conv_cuda(torch::Tensor input, torch::Tensor weight, int N, int C, int H, int W, int K, int G, int P) {
    auto output = torch::zeros({N, C / G, H, W}, input.options());
    dim3 block_size(1, 16, 16);
    dim3 grid_size(N * G * K, (H + 2 * P + block_size.y - 1) / block_size.y, (W + 2 * P + block_size.z - 1) / block_size.z);
    group_conv_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, K, G, P);
    return output;
}
"""

group_conv_cpp_source = "torch::Tensor group_conv_cuda(torch::Tensor input, torch::Tensor weight, int N, int C, int H, int W, int K, int G, int P);"

group_conv = load_inline(
    name='group_conv',
    cpp_sources=group_conv_cpp_source,
    cuda_sources=group_conv_source,
    functions=['group_conv_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)


class ShuffleNetUnitNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitNew, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shuffle = ChannelShuffle(groups)
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = relu

    def forward(self, x):
        out = self.relu.relu_cuda(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = self.relu.relu_cuda(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(1024, num_classes)
        self.relu = relu

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnitNew(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu.relu_cuda(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.relu.relu_cuda(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x