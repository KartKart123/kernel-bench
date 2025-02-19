import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tanh(x - bias)
tanh_sub_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tanh_sub_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    if (idx < total_elements) {
        int w = idx % W;
        int h = (idx / W) % H;
        int c = (idx / (H * W)) % C;
        int n = idx / (C * H * W);

        float val = x[idx] - bias[c];
        out[idx] = tanh(val);
    }
}

torch::Tensor tanh_sub_cuda(torch::Tensor x, torch::Tensor bias)
{
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto x_flat = x.contiguous();
    auto bias_flat = bias.contiguous();
    auto out = torch::zeros_like(x_flat);

    int total_elements = N * C * H * W;

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    tanh_sub_kernel<<<blocks, threads>>>(x_flat.data_ptr<float>(), bias_flat.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W);

    return out.view({N, C, H, W});
}
"""

tanh_sub_cpp_source = "torch::Tensor tanh_sub_cuda(torch::Tensor x, torch::Tensor bias);"

# Compile the inline CUDA code for tanh(x - bias)
tanh_sub = load_inline(
    name='tanh_sub',
    cpp_sources=[tanh_sub_cpp_source],
    cuda_sources=[tanh_sub_cuda_source],
    functions=['tanh_sub_cuda'],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.tanh_sub_cuda = tanh_sub.tanh_sub_cuda

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.tanh_sub_cuda(x, self.bias.view(-1))
        return x

batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]