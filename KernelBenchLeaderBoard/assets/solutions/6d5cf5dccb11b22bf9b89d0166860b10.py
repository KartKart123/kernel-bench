import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused addition and multiplication
fused_add_mul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_add_mul_kernel(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = x[idx] + y[idx];
        out[idx] = temp * y[idx];
    }
}

torch::Tensor fused_add_mul(torch::Tensor x, torch::Tensor y) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    fused_add_mul_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_add_mul_cpp_source = "torch::Tensor fused_add_mul(torch::Tensor x, torch::Tensor y);"

# Compile the inline CUDA code for fused addition and multiplication
fused_add_mul_op = load_inline(
    name='fused_add_mul',
    cpp_sources=fused_add_mul_cpp_source,
    cuda_sources=fused_add_mul_source,
    functions=['fused_add_mul'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernel for fused addition and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.fused_add_mul = fused_add_mul_op

    def forward(self, x, y):
        x = self.linear(x)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        x = self.fused_add_mul.fused_add_mul(x, y)
        return x