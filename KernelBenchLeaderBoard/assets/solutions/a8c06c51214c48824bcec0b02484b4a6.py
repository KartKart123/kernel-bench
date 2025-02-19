import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operations: matmul, divide, sum, and scale
fused_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(const float* x, const float* weight, float* out, int batch_size, int input_size, int hidden_size, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_size) {
        int batch_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;

        // Perform matrix multiplication
        float sum = 0.0;
        for (int i = 0; i < input_size; ++i) {
            sum += x[batch_idx * input_size + i] * weight[hidden_idx * input_size + i];
        }

        // Divide by 2
        sum /= 2.0;

        // Sum across the hidden dimension (since we are summing across the batch dimension)
        atomicAdd(&out[batch_idx], sum * scaling_factor);
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = weight.size(0);
    auto out = torch::zeros({batch_size, 1}, x.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), batch_size, input_size, hidden_size, scaling_factor
    );

    return out;
}
"""

fused_operations_cpp_source = "torch::Tensor fused_operations_cuda(torch::Tensor x, torch::Tensor weight, float scaling_factor);"

# Compile the inline CUDA code for the fused operations
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
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_operations = fused_operations

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        return self.fused_operations.fused_operations_cuda(x, self.weight, self.scaling_factor)