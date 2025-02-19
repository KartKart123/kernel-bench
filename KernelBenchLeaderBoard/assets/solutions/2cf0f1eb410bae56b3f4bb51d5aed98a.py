import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean reduction
mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_reduction_kernel(const float* x, float* out, int dim_size, int stride, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int batch_idx = idx / stride;
        int element_idx = idx % stride;
        float sum = 0.0;
        for (int i = 0; i < dim_size; ++i) {
            sum += x[batch_idx * dim_size * stride + i * stride + element_idx];
        }
        out[idx] = sum / dim_size;
    }
}

torch::Tensor mean_reduction_cuda(torch::Tensor x, int dim) {
    auto sizes = x.sizes();
    int dim_size = sizes[dim];
    int stride = 1;
    for (int i = dim + 1; i < sizes.size(); ++i) {
        stride *= sizes[i];
    }
    int num_elements = x.numel() / dim_size;
    auto out_sizes = sizes.vec();
    out_sizes.erase(out_sizes.begin() + dim);
    auto out = torch::empty(out_sizes, x.options());

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    mean_reduction_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), dim_size, stride, num_elements);

    return out;
}
"""

mean_reduction_cpp_source = "torch::Tensor mean_reduction_cuda(torch::Tensor x, int dim);"

# Compile the inline CUDA code for mean reduction
mean_reduction = load_inline(
    name='mean_reduction',
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=['mean_reduction_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)