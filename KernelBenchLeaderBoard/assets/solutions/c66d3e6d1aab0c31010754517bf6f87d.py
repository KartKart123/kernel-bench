import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused activation
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_activation_kernel(const float* __restrict__ x, float* __restrict__ y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Swish activation: val = val * sigmoid(val)
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        val = val * sigmoid_val;

        // Divide by 2.0
        val = val / 2.0f;

        // Clamp between -1.0 and 1.0
        val = fminf(fmaxf(val, -1.0f), 1.0f);

        // Tanh activation
        val = tanhf(val);

        // Clamp between -1.0 and 1.0 again
        val = fminf(fmaxf(val, -1.0f), 1.0f);

        y[idx] = val;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);

    int size = x.numel();
    const int threads = 256;
    const int blocks = (size + threads -1) / threads;

    fused_activation_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);

    return y;
}
"""

fused_activation_cpp_source = "torch::Tensor fused_activation_cuda(torch::Tensor x);"

# Compile the inline CUDA code for fused activation
fused_activation = load_inline(
    name='fused_activation',
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=['fused_activation_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    """
    Optimized Model with fused custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_activation.fused_activation_cuda(x)
        return x