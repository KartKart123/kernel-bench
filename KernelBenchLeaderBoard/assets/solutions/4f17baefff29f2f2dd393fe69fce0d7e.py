import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused operations
fused_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(const float* input, const float* multiply_weight, float* output, int size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int feature_idx = idx % out_features;
        float x = input[idx];
        float swish = x * 1.0f / (1.0f + expf(-x)); // Swish activation
        float result = swish * multiply_weight[feature_idx];
        output[idx] = result * 1.0f / (1.0f + expf(-result)); // Swish activation again
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor multiply_weight) {
    auto size = input.numel();
    auto out_features = multiply_weight.size(0);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), multiply_weight.data_ptr<float>(), output.data_ptr<float>(), size, out_features);

    return output;
}
"""

fused_operations_cpp_source = "torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor multiply_weight);"

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
    Optimized Model with custom CUDA kernel for fused operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.fused_operations = fused_operations

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.group_norm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.fused_operations.fused_operations_cuda(x, self.multiply_weight)
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