import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int batch_idx = idx / dim;
        int dim_idx = idx % dim;
        float cumsum = 0.0f;
        for (int i = 0; i <= dim_idx; i++) {
            if (mask[batch_idx * dim + i]) {
                cumsum += x[batch_idx * dim + i];
            }
        }
        out[idx] = cumsum;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    masked_cumsum_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mask.data_ptr<bool>(), out.data_ptr<float>(), size, dim);

    return out;
}
"""

masked_cumsum_cpp_source = (
    "torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask, int dim);"
)

masked_cumsum = load_inline(
    name="masked_cumsum",
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=["masked_cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum

    def forward(self, x, mask):
        return self.masked_cumsum.masked_cumsum_cuda(x, mask, x.shape[self.dim])