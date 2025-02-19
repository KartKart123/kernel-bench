import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operations
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ sum_tensor,
    float* __restrict__ y,
    int batch_size, int channels, int depth, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * depth * height * width;
    if (idx < total_elements)
    {
        int w = idx % width;
        int temp_idx = idx / width;
        int h = temp_idx % height;
        temp_idx = temp_idx / height;
        int d = temp_idx % depth;
        temp_idx = temp_idx / depth;
        int c = temp_idx % channels;
        int n = temp_idx / channels;

        // Get x value
        float x_val = x[idx];

        // Apply LeakyReLU
        float y_val = x_val >= 0.0f ? x_val : x_val * 0.2f;

        // Add sum_tensor[c]
        y_val += sum_tensor[c];

        // Clamp between -1.0 and 1.0
        y_val = fminf(fmaxf(y_val, -1.0f), 1.0f);

        // Apply GELU activation
        float gelu_in = y_val / 1.41421356237f;  // sqrt(2.0)
        float erf_val = erff(gelu_in);
        y_val = 0.5f * y_val * (1.0f + erf_val);

        // Write to output
        y[idx] = y_val;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor sum_tensor)
{
    // x: input tensor after convolution, shape (batch_size, channels, depth, height, width)
    // sum_tensor: tensor to add, shape (channels, 1, 1, 1)

    // Flatten sum_tensor to be 1D tensor with size [channels]
    auto sum_tensor_flat = sum_tensor.view({-1});

    // Allocate output tensor y
    auto y = torch::empty_like(x);

    // Get dimensions
    int batch_size = x.size(0);
    int channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);
    int total_elements = batch_size * channels * depth * height * width;

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        sum_tensor_flat.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size, channels, depth, height, width);

    return y;
}
"""

fused_op_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor x, torch::Tensor sum_tensor);"

# Compile the inline CUDA code for the fused operation
fused_op = load_inline(
    name='fused_op',
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=['fused_op_cuda'],
    verbose=False,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses LeakyReLU, addition, clamp, and GELU into a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_op = fused_op

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.fused_op_cuda(x, self.sum_tensor)
        return x