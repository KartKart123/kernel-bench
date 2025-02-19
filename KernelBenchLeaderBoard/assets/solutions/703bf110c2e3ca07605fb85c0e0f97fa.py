import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_activation_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_x = 1 / (1 + exp(-x[idx]));
        out[idx] = x[idx] * sigmoid_x;
    }
}

torch::Tensor swish_activation_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_activation_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

swish_activation_cpp_source = "torch::Tensor swish_activation_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Swish activation
swish_activation = load_inline(
    name='swish_activation',
    cpp_sources=swish_activation_cpp_source,
    cuda_sources=swish_activation_source,
    functions=['swish_activation_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_activation = swish_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_activation.swish_activation_cuda(x)