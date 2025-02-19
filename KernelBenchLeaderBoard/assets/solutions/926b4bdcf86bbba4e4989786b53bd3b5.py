import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Swish, Multiply, and Swish operations
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_swish_multiply_swish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ multiply_weight,
    float* __restrict__ out,
    int batch_size,
    int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    if (idx < total_elements) {
        int j = idx % out_features;
        float val = x[idx];
        val = val * sigmoidf(val);       // First Swish
        val = val * multiply_weight[j];  // Multiply by weight
        val = val * sigmoidf(val);       // Second Swish
        out[idx] = val;
    }
}

torch::Tensor fused_swish_multiply_swish_cuda(
    torch::Tensor x,
    torch::Tensor multiply_weight)
{
    int batch_size = x.size(0);
    int out_features = x.size(1);
    auto out = torch::zeros_like(x);
    int total_elements = batch_size * out_features;

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_swish_multiply_swish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        multiply_weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features);

    return out;
}
"""

cpp_source = """
torch::Tensor fused_swish_multiply_swish_cuda(torch::Tensor x, torch::Tensor multiply_weight);
"""

# Compile the inline CUDA code
fused_swish_multiply_swish = load_inline(
    name='fused_swish_multiply_swish',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['fused_swish_multiply_swish_cuda'],
    verbose=True
)

# Define the optimized model
class ModelNew(nn.Module):
    """
    Optimized Model using custom CUDA fused kernel for Swish and Multiply operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.fused_swish_multiply_swish = fused_swish_multiply_swish

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = self.fused_swish_multiply_swish.fused_swish_multiply_swish_cuda(x, self.multiply_weight)
        return x

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 16
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]