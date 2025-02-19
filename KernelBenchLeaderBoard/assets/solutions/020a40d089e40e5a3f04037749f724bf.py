import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for tanh activation
tanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int size = input.numel();

    const int threads = 512;
    const int blocks = (size + threads - 1) / threads;

    tanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor input);"

# Compile the inline CUDA code for tanh activation
tanh_activation = load_inline(
    name='tanh_activation',
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_cuda_source,
    functions=['tanh_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Model using custom CUDA tanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tanh_activation = tanh_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_activation.tanh_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed