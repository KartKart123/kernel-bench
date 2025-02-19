import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ChannelShuffle
channel_shuffle_cuda_source = """
#include <torch/extension.h>

__global__ void channel_shuffle_kernel(const float* input, float* output, int N, int C, int H, int W, int groups, int channels_per_group) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    if (index < total_elements) {
        int n = index / (C * H * W);
        int c = (index / (H * W)) % C;
        int h = (index / W) % H;
        int w = index % W;

        int g = c / channels_per_group;
        int cpg = c % channels_per_group;
        int c_out = cpg * groups + g;

        int output_index = n * C * H * W + c_out * H * W + h * W + w;
        output[output_index] = input[index];
    }
}

torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);
    int channels_per_group = C / groups;

    auto output = torch::empty_like(input);

    int total_elements = N * C * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    channel_shuffle_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, groups, channels_per_group
    );

    return output;
}
"""

channel_shuffle_cpp_source = """
torch::Tensor channel_shuffle_cuda(torch::Tensor input, int groups);
"""

# Load the custom CUDA code
channel_shuffle_module = load_inline(
    name='channel_shuffle',
    cpp_sources=channel_shuffle_cpp_source,
    cuda_sources=channel_shuffle_cuda_source,
    functions=['channel_shuffle_cuda'],
    verbose=False
)

class ChannelShuffleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, groups):
        ctx.groups = groups
        output = channel_shuffle_module.channel_shuffle_cuda(input, groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # The backward pass for channel shuffle is the same as the forward pass
        # because it's a permutation operation.
        groups = ctx.groups
        grad_input = channel_shuffle_module.channel_shuffle_cuda(grad_output, groups)
        return grad_input, None

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        return ChannelShuffleFunction.apply(x, self.groups)

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()

        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)

        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.shuffle(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out += self.shortcut(residual)
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

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def get_inputs():
    batch_size = 10
    input_channels = 3
    height = 224
    width = 224
    return [torch.randn(batch_size, input_channels, height, width).cuda()]

def get_init_inputs():
    num_classes = 1000
    return [num_classes]