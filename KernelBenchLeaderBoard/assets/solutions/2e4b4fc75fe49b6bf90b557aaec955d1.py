import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction
min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void min_reduction_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int dim_size, int stride, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int batch_idx = idx / stride;
        int element_idx = idx % stride;
        scalar_t min_val = input[batch_idx * dim_size * stride + element_idx];
        for (int i = 1; i < dim_size; ++i) {
            scalar_t val = input[batch_idx * dim_size * stride + i * stride + element_idx];
            if (val < min_val) {
                min_val = val;
            }
        }
        output[idx] = min_val;
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int dim_size = sizes[dim];
    int stride = 1;
    for (int i = dim + 1; i < sizes.size(); ++i) {
        stride *= sizes[i];
    }
    int num_elements = input.numel() / dim_size;

    auto output_sizes = sizes.vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int block_size = 256;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_reduction_cuda", ([&] {
        min_reduction_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            stride,
            num_elements
        );
    }));

    return output;
}
"""

min_reduction_cpp_source = "torch::Tensor min_reduction_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code for min reduction
min_reduction = load_inline(
    name='min_reduction',
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=['min_reduction_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return self.min_reduction.min_reduction_cuda(x, self.dim)