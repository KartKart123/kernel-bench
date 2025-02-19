import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_reduce_dim1_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int size0, int size1, int size2) {
    // input: [size0, size1, size2]
    // output: [size0, size2]
    int batch = blockIdx.x;
    int idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch < size0 && idx < size2) {
        scalar_t max_val = input[batch * size1 * size2 + 0 * size2 + idx];
        for (int i = 1; i < size1; ++i) {
            scalar_t val = input[batch * size1 * size2 + i * size2 + idx];
            if (val > max_val) {
                max_val = val;
            }
        }
        output[batch * size2 + idx] = max_val;
    }
}

torch::Tensor max_reduce_cuda(torch::Tensor input, int dim) {
    if (dim != 1) {
        throw std::runtime_error("Currently, only dim=1 is supported.");
    }
    auto sizes = input.sizes();
    int size0 = sizes[0];
    int size1 = sizes[1];
    int size2 = sizes[2];
    auto output = torch::empty({size0, size2}, input.options());

    const int threads = 256;
    const int blocks_x = size0;
    const int blocks_y = (size2 + threads - 1) / threads;

    dim3 blocks(blocks_x, blocks_y);
    dim3 threads_per_block(threads);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_reduce_cuda", ([&] {
        max_reduce_dim1_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size0, size1, size2);
    }));

    return output;
}
"""

cpp_source = """
torch::Tensor max_reduce_cuda(torch::Tensor input, int dim);
"""

max_reduce = load_inline(
    name='max_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['max_reduce_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduce = max_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return self.max_reduce.max_reduce_cuda(x.cuda(), self.dim)