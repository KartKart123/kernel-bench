import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation and bias addition
swish_add_bias_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_add_bias_kernel(const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ out, int total_elements, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int feature_idx = idx % features;
        float val = x[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        out[idx] = val * sigmoid_val + bias[feature_idx];
    }
}

torch::Tensor swish_add_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    auto total_elements = x.numel();
    auto features = x.size(1);
    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    swish_add_bias_kernel<<<blocks, threads>>>(x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), total_elements, features);

    return out;
}
"""

swish_add_bias_cpp_source = """
torch::Tensor swish_add_bias_cuda(torch::Tensor x, torch::Tensor bias);
"""

# Compile the inline CUDA code
swish_add_bias = load_inline(
    name='swish_add_bias',
    cpp_sources=swish_add_bias_cpp_source,
    cuda_sources=swish_add_bias_source,
    functions=['swish_add_bias_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.swish_add_bias = swish_add_bias

    def forward(self, x):
        x = self.matmul(x)
        x = self.swish_add_bias.swish_add_bias_cuda(x, self.bias)
        x = self.group_norm(x)
        return x