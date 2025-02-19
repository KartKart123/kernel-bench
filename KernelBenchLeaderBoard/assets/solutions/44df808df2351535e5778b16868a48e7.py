import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / (1.0f + expf(-x[idx]));
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

swish_cpp_source = "torch::Tensor swish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Swish activation
swish = load_inline(
    name='swish',
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=['swish_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x[idx] + 0.044715f * x[idx] * x[idx] * x[idx]))));
        out[idx] = x[idx] * cdf;
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for GELU activation
gelu = load_inline(
    name='gelu',
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=['gelu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.swish = swish
        self.gelu = gelu

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = self.swish.swish_cuda(x)
        x = torch.tanh(x)
        x = self.gelu.gelu_cuda(x)
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1)
        return x