import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the Sigmoid activation
sigmoid_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

torch::Tensor sigmoid_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    sigmoid_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

sigmoid_cuda_cpp_source = """
torch::Tensor sigmoid_cuda_forward(torch::Tensor input);
"""

# Compile the inline CUDA code for Sigmoid activation
sigmoid_cuda_extension = load_inline(
    name='sigmoid_cuda_extension',
    cpp_sources=sigmoid_cuda_cpp_source,
    cuda_sources=sigmoid_cuda_source,
    functions=['sigmoid_cuda_forward'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Sigmoid activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sigmoid_cuda = sigmoid_cuda_extension.sigmoid_cuda_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom CUDA Sigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return self.sigmoid_cuda(x)