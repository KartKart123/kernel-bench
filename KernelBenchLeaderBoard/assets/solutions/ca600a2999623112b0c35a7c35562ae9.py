import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N,
    int C,
    int H,
    int W,
    float constant_value,
    float scaling_factor)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = N * C * H * W;
    if (index < num_elements) {
        int c = (index / (H * W)) % C;
        float val = x[index];
        val = fminf(val, constant_value);
        val += bias[c];
        val *= scaling_factor;
        out[index] = val;
    }
}

torch::Tensor fused_cuda(torch::Tensor x, torch::Tensor bias, float constant_value, float scaling_factor) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty_like(x);

    int num_elements = N * C * H * W;
    const int threads = 1024;
    int blocks = (num_elements + threads -1) / threads;

    fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        constant_value,
        scaling_factor
    );
    return out;
}
"""

cpp_source = '''
torch::Tensor fused_cuda(torch::Tensor x, torch::Tensor bias, float constant_value, float scaling_factor);
'''

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_cuda'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA fused kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        bias = self.bias
        x = x.to(x.device)
        bias = bias.to(x.device)
        x = self.fused_ops.fused_cuda(x, bias, self.constant_value, self.scaling_factor)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
constant_value = 0.5
bias_shape = out_channels
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]