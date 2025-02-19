import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation and scaling
swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_scale_kernel(const float* x, float* out, float scaling_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[idx]));
        out[idx] = x[idx] * sigmoid_x * scaling_factor;
    }
}

torch::Tensor swish_scale_cuda(torch::Tensor x, float scaling_factor) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_scale_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), scaling_factor, size);

    return out;
}
"""

swish_scale_cpp_source = "torch::Tensor swish_scale_cuda(torch::Tensor x, float scaling_factor);"

# Compile the inline CUDA code for Swish activation and scaling
swish_scale = load_inline(
    name='swish_scale',
    cpp_sources=swish_scale_cpp_source,
    cuda_sources=swish_scale_source,
    functions=['swish_scale_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies Swish activation, and scales the result using custom CUDA kernels.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.swish_scale = swish_scale

    def forward(self, x):
        x = self.matmul(x)
        x = self.swish_scale.swish_scale_cuda(x, self.scaling_factor)
        return x