import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused logsumexp and ReLU
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void logsumexp_relu_kernel(const float* x, float* y, int N, int C, int D, int H, int W)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D * H * W;
    if (index >= total) return;

    int n = index / (D * H * W);
    int rest = index % (D * H * W);
    int d = rest / (H * W);
    rest = rest % (H * W);
    int h = rest / W;
    int w = rest % W;

    // First pass: compute max over c
    float m = -FLT_MAX;
    for (int c = 0; c < C; ++c)
    {
        int x_index = (((n * C + c) * D + d) * H + h) * W + w;
        float val = x[x_index];
        if (val > m)
            m = val;
    }

    // Second pass: compute sum over c
    float s = 0.0f;
    for (int c = 0; c < C; ++c)
    {
        int x_index = (((n * C + c) * D + d) * H + h) * W + w;
        float val = x[x_index];
        s += expf(val - m);
    }

    // Compute logsumexp and relu
    float lse = m + logf(s);
    float r = fmaxf(lse, 0.0f);

    // Write to output
    int y_index = ((n * D + d) * H + h) * W + w;
    y[y_index] = r;
}

torch::Tensor logsumexp_relu_cuda(torch::Tensor x)
{
    const int N = x.size(0);
    const int C = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);

    auto y = torch::zeros({N, 1, D, H, W}, x.options());

    const int total = N * D * H * W;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    logsumexp_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, D, H, W
    );

    return y;
}
"""

cpp_source = """
torch::Tensor logsumexp_relu_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
logsumexp_relu = load_inline(
    name='logsumexp_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['logsumexp_relu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized Model with custom CUDA operator for fused logsumexp and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.logsumexp_relu = logsumexp_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.logsumexp_relu.logsumexp_relu_cuda(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]