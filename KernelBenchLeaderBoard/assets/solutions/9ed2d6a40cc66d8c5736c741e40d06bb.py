import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activations and bias addition
fused_activation_bias_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ inline float gelu(float x) {
    // Approximate GELU implementation
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_activation_bias_kernel(const float* __restrict__ x,
                                             const float* __restrict__ bias,
                                             float* __restrict__ y,
                                             int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * D * H * W;
    if (idx >= total_elements) return;

    // Calculate channel index
    int tmp = idx / (D * H * W);
    int c = tmp % C;

    float val = x[idx];
    val = relu(val);
    val = gelu(val);
    val = sigmoid(val);
    val += bias[c];  // bias shape is (C, 1, 1, 1), broadcast over dimensions

    y[idx] = val;
}

torch::Tensor fused_activation_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    auto N = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);

    auto y = torch::empty_like(x);

    int total_elements = N * C * D * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_activation_bias_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, D, H, W
    );
    return y;
}
"""

fused_activation_bias_cpp_source = """
torch::Tensor fused_activation_bias_cuda(torch::Tensor x, torch::Tensor bias);
"""

# Compile the inline CUDA code for fused activation and bias addition
fused_activation_bias = load_inline(
    name='fused_activation_bias',
    cpp_sources=fused_activation_bias_cpp_source,
    cuda_sources=fused_activation_bias_source,
    functions=['fused_activation_bias_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with fused custom CUDA kernel for activations and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_activation_bias = fused_activation_bias

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_activation_bias.fused_activation_bias_cuda(x, self.bias)
        return x