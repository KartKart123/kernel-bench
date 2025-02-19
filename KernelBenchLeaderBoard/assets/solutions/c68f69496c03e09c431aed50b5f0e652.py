import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for ELU activation
elu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elu_forward_kernel(const float* __restrict__ input, float* __restrict__ output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0 ? x : alpha * (expf(x) - 1.0f);
    }
}

torch::Tensor elu_forward_cuda(torch::Tensor input, float alpha) {
    int size = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;

    elu_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), alpha, size);

    return output;
}
"""

# C++ header declarations
elu_cpp_source = """
torch::Tensor elu_forward_cuda(torch::Tensor input, float alpha);
"""

# Load the inline CUDA code for ELU activation
elu_extension = load_inline(
    name='elu_extension',
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_cuda_source,
    functions=['elu_forward_cuda'],
    verbose=False,
    extra_cflags=[],
    extra_cuda_cflags=[],
    extra_ldflags=[],
    extra_include_paths=[],
)

class ModelNew(nn.Module):
    """
    Optimized ELU activation model with custom CUDA kernel.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the optimized ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_extension = elu_extension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the custom ELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with custom ELU applied, same shape as input.
        """
        if not x.is_cuda:
            x = x.cuda()
        return self.elu_extension.elu_forward_cuda(x, self.alpha)