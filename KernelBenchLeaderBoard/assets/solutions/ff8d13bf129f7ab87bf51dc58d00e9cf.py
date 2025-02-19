import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for product reduction over a dimension
cuda_source = """
#include <torch/types.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void prod_dim_kernel(const scalar_t* __restrict__ input,
                                scalar_t* __restrict__ output,
                                int64_t dim_size,
                                int64_t outer_size,
                                int64_t inner_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_threads = gridDim.x * blockDim.x;
    int64_t out_elements = outer_size * inner_size;

    for (int64_t i = idx; i < out_elements; i += total_threads) {
        int64_t outer_idx = i / inner_size;
        int64_t inner_idx = i % inner_size;

        scalar_t prod = 1.0;

        for (int64_t d = 0; d < dim_size; ++d) {
            int64_t input_idx = outer_idx * dim_size * inner_size + d * inner_size + inner_idx;
            prod *= input[input_idx];
        }
        output[i] = prod;
    }
}

torch::Tensor prod_dim(torch::Tensor input, int64_t dim) {
    // Ensure input is contiguous and on CUDA
    input = input.contiguous();
    if (!input.is_cuda()) {
        AT_ERROR("Input must be a CUDA tensor");
    }

    auto sizes = input.sizes();
    int64_t ndim = sizes.size();
    dim = dim < 0 ? dim + ndim : dim;

    // Compute sizes for kernel launch
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    int64_t dim_size = sizes[dim];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= sizes[i];
    }

    // Output tensor
    std::vector<int64_t> output_sizes;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_sizes.push_back(sizes[i]);
        }
    }
    auto output = torch::empty(output_sizes, input.options());

    int64_t output_elements = output.numel();

    const int threads = 256;
    const int blocks = (output_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prod_dim_kernel", ([&] {
        prod_dim_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            outer_size,
            inner_size);
    }));
    cudaDeviceSynchronize();

    return output;
}
"""

cpp_source = """
torch::Tensor prod_dim(torch::Tensor input, int64_t dim);
"""

# Compile the inline CUDA code
prod_dim_module = load_inline(
    name='prod_dim_module',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['prod_dim'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Simple model that performs product reduction over a dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.prod_dim = prod_dim_module.prod_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prod_dim(x, self.dim)