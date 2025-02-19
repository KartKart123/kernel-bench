import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_ops_kernel(const float* x, float* out, int size, float subtract_value, float multiply_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = x[idx] - subtract_value;
        out[idx] = max(0.0f, temp * multiply_value);
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x, float subtract_value, float multiply_value) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_ops_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, subtract_value, multiply_value);

    return out;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor x, float subtract_value, float multiply_value);"

# Compile the inline CUDA code for fused operations
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=['fused_ops_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_ops.fused_ops_cuda(x, self.subtract_value, self.multiply_value)
        return x