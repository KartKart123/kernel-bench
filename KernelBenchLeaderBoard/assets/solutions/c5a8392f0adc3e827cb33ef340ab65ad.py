import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Tanh activation
tanh_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void tanh_activation_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(x[idx]);
    }
}

torch::Tensor tanh_activation_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    tanh_activation_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

tanh_activation_cpp_source = (
    "torch::Tensor tanh_activation_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Tanh activation
tanh_activation = load_inline(
    name="tanh_activation",
    cpp_sources=tanh_activation_cpp_source,
    cuda_sources=tanh_activation_source,
    functions=["tanh_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tanh_activation = tanh_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh_activation.tanh_activation_cuda(x)