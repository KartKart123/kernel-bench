import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fused operations
fused_linear_swish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float swish(float x) {
    return x * (__fdividef(1.0f, 1.0f + expf(-x)));
}

__global__ void fused_linear_swish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int batch_size,
    int in_features,
    int out_features,
    float scaling_factor)
{
    // Compute y = scaling_factor * swish( x @ W^T + b )

    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // out_features index

    if (row < batch_size && col < out_features) {
        float z = 0.0f;
        // Compute dot product of x[row, :] and weight[col, :]
        for (int k = 0; k < in_features; ++k) {
            z += x[row * in_features + k] * weight[col * in_features + k];
        }
        // Add bias
        z += bias[col];
        // Apply Swish activation and scaling
        float y_ij = scaling_factor * swish(z);
        y[row * out_features + col] = y_ij;
    }
}

torch::Tensor fused_linear_swish_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float scaling_factor) {
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto y = torch::empty({batch_size, out_features}, x.options());

    const int threads = 16;
    dim3 threads_per_block(threads, threads);
    dim3 num_blocks( (out_features + threads - 1) / threads,
                     (batch_size + threads - 1) / threads);

    fused_linear_swish_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        scaling_factor);

    return y;
}
"""

fused_linear_swish_cpp_source = """
torch::Tensor fused_linear_swish_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float scaling_factor);
"""

# Compile the inline CUDA code for fused operation
fused_linear_swish = load_inline(
    name='fused_linear_swish',
    cpp_sources=fused_linear_swish_cpp_source,
    cuda_sources=fused_linear_swish_source,
    functions=['fused_linear_swish_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.fused_op = fused_linear_swish

    def forward(self, x):
        x = x.contiguous()
        return self.fused_op.fused_linear_swish_cuda(x, self.weight, self.bias, self.scaling_factor)

batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]