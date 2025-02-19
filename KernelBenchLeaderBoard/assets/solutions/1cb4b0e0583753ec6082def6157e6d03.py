import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cumulative product
cumprod_source = """
#include <torch/extension.h>

__global__ void cumprod_kernel(const float* x, float* y, int64_t length) {
    int batch_idx = blockIdx.x;
    const float* x_batch = x + batch_idx * length;
    float* y_batch = y + batch_idx * length;

    float cumprod = 1.0f;
    for (int64_t i = 0; i < length; ++i) {
        cumprod *= x_batch[i];
        y_batch[i] = cumprod;
    }
}

torch::Tensor cumprod_cuda(torch::Tensor x, int64_t dim) {
    // Ensure input tensor is on CUDA and is contiguous
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 2 && dim == 1, "Currently only supports 2D tensors with dim=1");

    int64_t batch_size = x.size(0);
    int64_t length = x.size(1);

    auto y = torch::empty_like(x);

    // Launch one kernel per batch
    cumprod_kernel<<<batch_size, 1>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        length
    );

    return y;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda(torch::Tensor x, int64_t dim);
"""

# Compile the inline CUDA code for cumulative product
cumprod = load_inline(
    name='cumprod',
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_source,
    functions=['cumprod_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_cuda = cumprod.cumprod_cuda

    def forward(self, x):
        return self.cumprod_cuda(x, self.dim)

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]