import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition, division, and Swish activation
fused_bias_div_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bias_div_swish_kernel(const float* x, float* out, float bias, float divide_value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = (x[idx] + bias) / divide_value;
        float sigmoid_temp = 1.0f / (1.0f + expf(-temp));
        out[idx] = temp * sigmoid_temp;
    }
}

torch::Tensor fused_bias_div_swish_cuda(torch::Tensor x, float bias, float divide_value) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    fused_bias_div_swish_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), bias, divide_value, size);

    return out;
}
"""

fused_bias_div_swish_cpp_source = "torch::Tensor fused_bias_div_swish_cuda(torch::Tensor x, float bias, float divide_value);"

# Compile the inline CUDA code
fused_bias_div_swish = load_inline(
    name='fused_bias_div_swish',
    cpp_sources=fused_bias_div_swish_cpp_source,
    cuda_sources=fused_bias_div_swish_source,
    functions=['fused_bias_div_swish_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized Model that uses a custom CUDA kernel to fuse bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_bias_div_swish = fused_bias_div_swish

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        x = self.fused_bias_div_swish.fused_bias_div_swish_cuda(x, self.bias.item(), self.divide_value)
        return x