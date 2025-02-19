import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* x, float* out, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (x[idx] > 0) ? x[idx] : x[idx] * negative_slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, negative_slope);

    return out;
}
"""

leaky_relu_cpp_source = "torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope);"

# Compile the inline CUDA code for LeakyReLU
leaky_relu = load_inline(
    name='leaky_relu',
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=['leaky_relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation using custom CUDA kernel.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leaky_relu = leaky_relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return self.leaky_relu.leaky_relu_cuda(x, self.negative_slope)