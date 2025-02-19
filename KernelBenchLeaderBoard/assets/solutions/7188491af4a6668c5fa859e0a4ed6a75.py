import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused Conv1 + BatchNorm1 + ReLU kernel
__global__ void fused_conv1_bn_relu_kernel(
    const float* input, const float* weight, const float* bn_weight,
    const float* bn_bias, const float* bn_mean, const float* bn_var,
    float* output, int batch_size, int in_channels, int out_channels,
    int height, int width, int groups) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width) return;
    
    int w_idx = idx % width;
    int h_idx = (idx / width) % height;
    int c_idx = (idx / (width * height)) % out_channels;
    int b_idx = idx / (width * height * out_channels);
    
    int channels_per_group = in_channels / groups;
    int group = c_idx / (out_channels / groups);
    
    float sum = 0.0f;
    for(int ic = group * channels_per_group; ic < (group + 1) * channels_per_group; ic++) {
        sum += input[b_idx * in_channels * height * width + 
                    ic * height * width +
                    h_idx * width + w_idx] *
               weight[c_idx * channels_per_group * 1 * 1 + 
                     (ic - group * channels_per_group)];
    }
    
    // Apply BatchNorm + ReLU
    float bn_output = (sum - bn_mean[c_idx]) / sqrtf(bn_var[c_idx] + 1e-5f);
    bn_output = bn_output * bn_weight[c_idx] + bn_bias[c_idx];
    output[idx] = bn_output > 0 ? bn_output : 0;
}

// Channel Shuffle kernel
__global__ void channel_shuffle_kernel(
    const float* input, float* output,
    int batch_size, int channels, int height, int width, int groups) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width) return;
    
    int w_idx = idx % width;
    int h_idx = (idx / width) % height;
    int c_idx = (idx / (width * height)) % channels;
    int b_idx = idx / (width * height * channels);
    
    int channels_per_group = channels / groups;
    int group = c_idx / channels_per_group;
    int channel_in_group = c_idx % channels_per_group;
    
    int new_c_idx = channel_in_group * groups + group;
    
    output[b_idx * channels * height * width +
           new_c_idx * height * width +
           h_idx * width + w_idx] = 
        input[b_idx * channels * height * width +
              c_idx * height * width +
              h_idx * width + w_idx];
}

torch::Tensor fused_conv1_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    int groups) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1); 
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_channels = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                             input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * height * width + threads - 1) / threads;
    
    fused_conv1_bn_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, height, width, groups);
    
    return output;
}

torch::Tensor channel_shuffle_cuda(
    torch::Tensor input, int groups) {
    
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (batch_size * channels * height * width + threads - 1) / threads;
    
    channel_shuffle_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, height, width, groups);
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_conv1_bn_relu_cuda(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    int groups);

torch::Tensor channel_shuffle_cuda(
    torch::Tensor input, int groups);
"""

custom_ops = load_inline(
    name='custom_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_conv1_bn_relu_cuda', 'channel_shuffle_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.groups = groups
        
        # First group conv + bn + relu (fused)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise conv + bn
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False) 
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second group conv + bn
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        # Fused conv1 + bn1 + relu
        out = custom_ops.fused_conv1_bn_relu_cuda(
            x, self.conv1.weight,
            self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var,
            self.groups)
        
        out = self.bn2(self.conv2(out))
        
        # Custom channel shuffle
        out = custom_ops.channel_shuffle_cuda(out, self.groups)
        
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        return out