import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for masked cumulative sum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float cumsum = 0.0;
        for (int i = 0; i < stride; ++i) {
            int index = idx * stride + i;
            if (mask[index]) {
                cumsum += x[index];
            }
            out[index] = cumsum;
        }
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    auto size = x.size(0);
    auto stride = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    masked_cumsum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mask.data_ptr<bool>(), out.data_ptr<float>(), size, stride);

    return out;
}
"""

masked_cumsum_cpp_source = "torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);"

# Compile the inline CUDA code for masked cumulative sum
masked_cumsum = load_inline(
    name='masked_cumsum',
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=['masked_cumsum_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum

    def forward(self, x, mask):
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, self.dim)