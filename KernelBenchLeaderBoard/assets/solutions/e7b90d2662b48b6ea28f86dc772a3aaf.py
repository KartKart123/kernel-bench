import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for group convolution
group_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_conv_kernel(const float* input, const float* weight, float* output, 
                                  int batch_size, int in_channels, int out_channels, int height, int width, 
                                  int kernel_size, int stride, int padding, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    
    int channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    for (int i = idx; i < batch_size * out_channels * height * width; i += num_threads) {
        int b = i / (out_channels * height * width);
        int oc = (i / (height * width)) % out_channels;
        int h = (i / width) % height;
        int w = i % width;
        
        int group_idx = oc / out_channels_per_group;
        int oc_in_group = oc % out_channels_per_group;
        
        float sum = 0.0f;
        for (int ic = 0; ic < channels_per_group; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = h * stride + kh - padding;
                    int iw = w * stride + kw - padding;
                    
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        int input_idx = b * in_channels * height * width + (group_idx * channels_per_group + ic) * height * width + ih * width + iw;
                        int weight_idx = group_idx * out_channels_per_group * channels_per_group * kernel_size * kernel_size + 
                                         oc_in_group * channels_per_group * kernel_size * kernel_size + 
                                         ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[i] = sum;
    }
}

torch::Tensor group_conv_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;
    
    group_conv_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), 
                                                 batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, groups);
    
    return output;
}
"""

group_conv_cpp_source = "torch::Tensor group_conv_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int groups);"

# Compile the inline CUDA code for group convolution
group_conv = load_inline(
    name='group_conv',
    cpp_sources=group_conv_cpp_source,
    cuda_sources=group_conv_source,
    functions=['group_conv_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()
        
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
        
        self.group_conv = group_conv
    
    def forward(self, x):
        out = F.relu(self.bn1(self.group_conv.group_conv_cuda(x, self.conv1.weight, 1, 0, self.conv1.groups)))
        out = self.bn2(self.group_conv.group_conv_cuda(out, self.conv2.weight, 1, 1, self.conv2.groups))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.group_conv.group_conv_cuda(out, self.conv3.weight, 1, 0, self.conv3.groups)))
        
        out += self.shortcut(x)
        return out

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