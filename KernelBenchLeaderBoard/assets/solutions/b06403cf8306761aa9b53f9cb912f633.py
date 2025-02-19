import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_x = 1.0f / (1.0f + expf(-x[idx]));
        out[idx] = x[idx] * sigmoid_x;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

swish_cpp_source = "torch::Tensor swish_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Swish activation
swish = load_inline(
    name='swish',
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=['swish_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.swish = swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish.swish_cuda(x)