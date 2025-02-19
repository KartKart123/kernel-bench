import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l2_norm_kernel(const float* x, float* out, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += x[idx * dim + i] * x[idx * dim + i];
        }
        sum = sqrtf(sum);
        for (int i = 0; i < dim; i++) {
            out[idx * dim + i] = x[idx * dim + i] / sum;
        }
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    l2_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for L2 normalization
l2_norm = load_inline(
    name="l2_norm",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_norm.l2_norm_cuda(x)