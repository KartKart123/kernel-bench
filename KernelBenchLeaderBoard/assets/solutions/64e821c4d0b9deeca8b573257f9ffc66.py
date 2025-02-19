import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel
custom_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_op_kernel(const float* x, const float* m, float* y, int N, int C, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    if (index < total_elements) {
        // Compute indices
        int n = index / (C * H * W);
        int c = (index / (H * W)) % C;
        int h = (index / W) % H;
        int w = index % W;

        // Flattened index
        int x_index = ((n * C + c) * H + h) * W + w;
        int m_index = c; // m is of shape [C, 1, 1]

        float val = x[x_index] * m[m_index];

        // Apply LeakyReLU
        const float negative_slope = 0.01f;
        val = (val > 0) ? val : val * negative_slope;

        // Apply GELU using tanh approximation
        const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
        const float coeff = 0.044715f;
        float tanh_arg = sqrt_2_over_pi * (val + coeff * val * val * val);
        val = 0.5f * val * (1.0f + tanhf(tanh_arg));

        y[x_index] = val;
    }
}

torch::Tensor custom_op_cuda(torch::Tensor x, torch::Tensor m) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    auto y = torch::empty_like(x);

    int total_elements = N * C * H * W;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    custom_op_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), m.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W);

    return y;
}
"""

custom_op_cpp_source = "torch::Tensor custom_op_cuda(torch::Tensor x, torch::Tensor m);"

# Compile the inline CUDA code
custom_op = load_inline(
    name='custom_op',
    cpp_sources=custom_op_cpp_source,
    cuda_sources=custom_op_source,
    functions=['custom_op_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized Model that replaces multiplication, LeakyReLU, and GELU with a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.custom_op = custom_op

    def forward(self, x):
        x = self.conv(x)
        x = self.custom_op.custom_op_cuda(x, self.multiplier)
        return x