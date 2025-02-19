import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumulative_product_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumulative_product_kernel(const float* x, float* out, int batch_size, int input_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float product = 1.0f;
        for (int i = 0; i < input_size; i++) {
            int index = idx * input_size + i;
            product *= x[index];
            out[index] = product;
        }
    }
}

torch::Tensor cumulative_product_cuda(torch::Tensor x, int dim) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    cumulative_product_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, input_size, dim);

    return out;
}
"""

cumulative_product_cpp_source = "torch::Tensor cumulative_product_cuda(torch::Tensor x, int dim);"

cumulative_product = load_inline(
    name='cumulative_product',
    cpp_sources=cumulative_product_cpp_source,
    cuda_sources=cumulative_product_source,
    functions=['cumulative_product_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumulative_product = cumulative_product

    def forward(self, x):
        return self.cumulative_product.cumulative_product_cuda(x, self.dim)