import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l1_norm_kernel(const float* x, float* out, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += fabs(x[idx * dim + i]);
        }
        for (int i = 0; i < dim; i++) {
            out[idx * dim + i] = x[idx * dim + i] / sum;
        }
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    l1_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name='l1_norm',
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=['l1_norm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.l1_norm_cuda(x)