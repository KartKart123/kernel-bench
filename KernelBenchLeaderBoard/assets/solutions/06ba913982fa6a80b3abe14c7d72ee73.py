import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_op_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scaling_factor,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int D, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D * H * W;
    if (idx >= total_elements) return;

    // Compute indices
    int temp = idx;
    int w = temp % W;
    temp = temp / W;
    int h = temp % H;
    temp = temp / H;
    int d = temp % D;
    temp = temp / D;
    int c = temp % C;
    int n = temp / C;

    // Compute the offset into the x tensor
    int x_offset = ((n * C + c) * D + d) * H * W + h * W + w;

    // Get x value
    float x_val = x[x_offset];

    // Get scaling_factor and bias for channel c
    float scaling_factor_val = scaling_factor[c];
    float bias_val = bias[c];

    // Compute the fused operation
    float out_val = x_val * scaling_factor_val; // x * scaling_factor
    out_val = tanhf(out_val);                   // tanh(x)
    out_val = out_val * bias_val;               // x * bias
    out_val = 1.0f / (1.0f + expf(-out_val));   // sigmoid(x)

    // Write the result
    out[x_offset] = out_val;
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias)
{
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int total_elements = N * C * D * H * W;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_op_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        scaling_factor.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W);

    return out;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor scaling_factor, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name='fused_op',
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=['fused_op_cuda'],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernel to fuse operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(out_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        # Squeeze scaling_factor and bias to shape (C,)
        scaling_factor_flat = self.scaling_factor.view(-1)
        bias_flat = self.bias.view(-1)
        x = self.fused_op.fused_op_cuda(x, scaling_factor_flat, bias_flat)
        return x