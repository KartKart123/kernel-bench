import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for combined LeakyReLU, multiplication, and LeakyReLU
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void custom_leaky_relu_mul_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ multiplier, 
    float* __restrict__ out, 
    int N, int C, int D, int H, int W, float negative_slope) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D * H * W;
    if (idx < total_elements) {
        int w = idx % W;
        int h = (idx / W) % H;
        int d = (idx / (W * H)) % D;
        int c = (idx / (W * H * D)) % C;
        int n = idx / (W * H * D * C);

        // Compute flat index for x and out
        int x_index = (((n * C + c) * D + d) * H + h) * W + w;
        // multiplier has shape (C, 1, 1, 1), so we index by c
        float val = x[x_index];
        // First LeakyReLU
        val = (val >= 0) ? val : val * negative_slope;
        // Multiply by multiplier[c]
        float mul = multiplier[c];
        val = val * mul;
        // Second LeakyReLU
        val = (val >= 0) ? val : val * negative_slope;
        out[x_index] = val;
    }
}

torch::Tensor custom_leaky_relu_mul(torch::Tensor x, torch::Tensor multiplier, float negative_slope) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);

    auto total_elements = N * C * D * H * W;

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    custom_leaky_relu_mul_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W, negative_slope
    );

    return out;
}
"""

cpp_source = """
torch::Tensor custom_leaky_relu_mul(torch::Tensor x, torch::Tensor multiplier, float negative_slope);
"""

# Compile the inline CUDA code
custom_op = load_inline(
    name='custom_leaky_relu_mul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['custom_leaky_relu_mul'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized Model that fuses two LeakyReLU activations and a multiplication by a learnable parameter
    into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.negative_slope = 0.2
        self.custom_op = custom_op

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.custom_op.custom_leaky_relu_mul(x, self.multiplier, self.negative_slope)
        x = self.max_pool(x)
        return x