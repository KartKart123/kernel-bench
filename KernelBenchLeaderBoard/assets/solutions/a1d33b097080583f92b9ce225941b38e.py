import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the optimized operations
custom_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(const float* input, float* output, int input_size, int hidden_size, float scale_factor, float clamp_min, float clamp_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        float val = input[idx] * scale_factor;
        val = val + val;  // Residual connection
        val = fminf(fmaxf(val, clamp_min), clamp_max);  // Clamp
        output[idx] = val;
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input, float scale_factor, float clamp_min, float clamp_max) {
    auto hidden_size = input.size(1);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (hidden_size + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), hidden_size, scale_factor, clamp_min, clamp_max);

    return output;
}
"""

custom_kernel_cpp_source = "torch::Tensor fused_operations_cuda(torch::Tensor input, float scale_factor, float clamp_min, float clamp_max);"

# Compile the inline CUDA code for the fused operations
fused_operations = load_inline(
    name='fused_operations',
    cpp_sources=custom_kernel_cpp_source,
    cuda_sources=custom_kernel_source,
    functions=['fused_operations_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_operations = fused_operations

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.matmul(x)
        x = self.fused_operations.fused_operations_cuda(x, self.scale_factor, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)  # Mish activation
        return x

batch_size = 128
input_size = 512
hidden_size = 1024
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]