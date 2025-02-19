import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l1_normalization_kernel(const float* x, float* out, float* sum_abs, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int b = idx / dim;
        int d = idx % dim;
        atomicAdd(&sum_abs[b], fabsf(x[idx]));
    }

    __syncthreads();

    if (idx < batch_size * dim) {
        int b = idx / dim;
        int d = idx % dim;
        out[idx] = x[idx] / sum_abs[b];
    }
}

std::vector<torch::Tensor> l1_normalization_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto out = torch::zeros_like(x);
    auto sum_abs = torch::zeros(batch_size, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    const int block_size = 256;
    const int num_blocks = (batch_size * dim + block_size - 1) / block_size;

    l1_normalization_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), sum_abs.data_ptr<float>(), batch_size, dim);

    return {out};
}
"""

l1_normalization_cpp_source = "std::vector<torch::Tensor> l1_normalization_cuda(torch::Tensor x);"

# Compile the inline CUDA code for L1 normalization
l1_normalization = load_inline(
    name='l1_normalization',
    cpp_sources=l1_normalization_cpp_source,
    cuda_sources=l1_normalization_source,
    functions=['l1_normalization_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1_normalization = l1_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_normalization.l1_normalization_cuda(x)[0]