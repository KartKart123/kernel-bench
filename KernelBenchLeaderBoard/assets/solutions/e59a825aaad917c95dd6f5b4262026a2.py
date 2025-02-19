import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(const float* matmul_out, const float* bias, float* out, int size, float divide_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = matmul_out[idx] + bias[0];
        x = x / divide_value;
        out[idx] = x * 1.0f / (1.0f + expf(-x));  // Swish activation
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor matmul_out, torch::Tensor bias, float divide_value) {
    auto size = matmul_out.numel();
    auto out = torch::zeros_like(matmul_out);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(matmul_out.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), size, divide_value);

    return out;
}
"""

fused_operations_cpp_source = "torch::Tensor fused_operations_cuda(torch::Tensor matmul_out, torch::Tensor bias, float divide_value);"

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
    Optimized model with custom CUDA kernel for fused operations.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_operations = fused_operations

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        x = self.fused_operations.fused_operations_cuda(x, self.bias, self.divide_value)
        return x

batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]