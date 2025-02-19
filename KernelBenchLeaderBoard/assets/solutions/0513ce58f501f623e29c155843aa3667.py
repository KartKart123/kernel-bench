import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean over spatial dimensions and bias addition
mean_bias_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_bias_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ m, int batch_size, int channels, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels;
    if (idx < total) {
        int b = idx / channels;
        int c = idx % channels;

        float sum = 0.0f;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                sum += x[((b * channels + c) * H + h) * W + w];
            }
        }
        float mean = sum / (H * W);
        m[idx] = mean + bias[c];
    }
}

torch::Tensor mean_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    int batch_size = x.size(0);
    int channels = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto m = torch::empty({batch_size, channels}, x.options());

    int total = batch_size * channels;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    mean_bias_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), m.data_ptr<float>(), batch_size, channels, H, W);

    return m;
}
"""

mean_bias_cpp_source = "torch::Tensor mean_bias_cuda(torch::Tensor x, torch::Tensor bias);"

# Compile the inline CUDA code for mean over spatial dimensions and bias addition
mean_bias = load_inline(
    name='mean_bias',
    cpp_sources=mean_bias_cpp_source,
    cuda_sources=mean_bias_cuda_source,
    functions=['mean_bias_cuda'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.mean_bias = mean_bias

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use custom CUDA kernel for mean over spatial dimensions and bias addition
        x = self.mean_bias.mean_bias_cuda(x, self.bias.view(-1))
        # x is now of shape (batch_size, out_channels)
        # Apply logsumexp over channels (dim=1)
        x = torch.logsumexp(x, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        x = x * 10.0
        return x