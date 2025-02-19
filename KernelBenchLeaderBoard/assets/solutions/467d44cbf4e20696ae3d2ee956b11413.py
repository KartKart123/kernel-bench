import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int out_channels, int depth, int height, int width, float divisor) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * depth * height * width;

    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int d = (idx / (width * height)) % depth;
        int c = (idx / (width * height * depth)) % out_channels;
        int b = idx / (width * height * depth * out_channels);

        // Perform the operations: divide, add bias
        float val = input[idx] / divisor + bias[c];
        output[idx] = val;
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias, float divisor) {
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth * height * width + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, depth, height, width, divisor
    );

    return output;
}
"""

fused_operations_cpp_source = "torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias, float divisor);"

# Compile the inline CUDA code for fused operations
fused_operations = load_inline(
    name='fused_operations',
    cpp_sources=fused_operations_cpp_source,
    cuda_sources=fused_operations_source,
    functions=['fused_operations_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized Model with custom CUDA kernel for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.fused_operations = fused_operations

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = self.fused_operations.fused_operations_cuda(x, self.bias, self.divisor)
        x = torch.sum(x, dim=self.sum_dim)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]