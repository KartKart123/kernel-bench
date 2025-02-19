import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the C++ interface
fused_kernel_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_kernel_cuda(torch::Tensor x_conv, torch::Tensor x_norm);
"""

# Define the CUDA source
fused_kernel_cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_kernel(const float* x_conv, const float* x_norm, float* x_logsumexp,
                             int batch_size, int channels, int height, int width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * height * width;

    if (index < total_elements)
    {
        int w = index % width;
        int h = (index / width) % height;
        int b = index / (height * width);

        // Compute LogSumExp over channels for this (b, h, w)
        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; ++c)
        {
            int idx = ((b * channels + c) * height + h) * width + w;
            float x_norm_val = x_norm[idx];

            // Tanh
            float x_tanh = tanhf(x_norm_val);

            // Hardswish
            float x_tmp = x_tanh + 3.0f;
            x_tmp = fminf(fmaxf(x_tmp, 0.0f), 6.0f);
            float x_hardswish = x_tanh * x_tmp / 6.0f;

            // Residual addition
            float x_conv_val = x_conv[idx];
            float x_res = x_conv_val + x_hardswish;

            if (x_res > max_val)
                max_val = x_res;
        }

        // Compute sum of exp(x_res - max_val)
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; ++c)
        {
            int idx = ((b * channels + c) * height + h) * width + w;
            float x_norm_val = x_norm[idx];

            // Tanh
            float x_tanh = tanhf(x_norm_val);

            // Hardswish
            float x_tmp = x_tanh + 3.0f;
            x_tmp = fminf(fmaxf(x_tmp, 0.0f), 6.0f);
            float x_hardswish = x_tanh * x_tmp / 6.0f;

            // Residual addition
            float x_conv_val = x_conv[idx];
            float x_res = x_conv_val + x_hardswish;

            sum_exp += expf(x_res - max_val);
        }

        float logsumexp = max_val + logf(sum_exp);

        // Store output
        int out_idx = (b * height + h) * width + w; // Output has size (batch_size, 1, height, width)
        x_logsumexp[out_idx] = logsumexp;
    }
}

torch::Tensor fused_kernel_cuda(torch::Tensor x_conv, torch::Tensor x_norm)
{
    const auto batch_size = x_conv.size(0);
    const auto channels = x_conv.size(1);
    const auto height = x_conv.size(2);
    const auto width = x_conv.size(3);

    auto x_logsumexp = torch::zeros({batch_size, 1, height, width}, x_conv.options());

    const int threads = 1024;
    const int blocks = (batch_size * height * width + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(x_conv.data_ptr<float>(), x_norm.data_ptr<float>(), x_logsumexp.data_ptr<float>(),
                                      batch_size, channels, height, width);

    return x_logsumexp;
}
'''

# Compile the inline CUDA code
fused_kernel = load_inline(
    name='fused_kernel',
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_cuda_source,
    functions=['fused_kernel_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    """
    Optimized Model with fused custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.fused_kernel = fused_kernel

    def forward(self, x):
        x_conv = self.conv(x)  # Convolution
        x_norm = self.group_norm(x_conv)  # Group Normalization
        x_logsumexp = self.fused_kernel.fused_kernel_cuda(x_conv, x_norm)  # Fused Tanh, HardSwish, Residual Addition, LogSumExp
        return x_logsumexp

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]