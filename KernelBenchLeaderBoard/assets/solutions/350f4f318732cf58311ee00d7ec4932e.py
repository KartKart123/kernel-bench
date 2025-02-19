import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_reduce_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_reduce_kernel(const scalar_t* input, scalar_t* output, 
                                int batch_size, int dim1, int dim2, int reduce_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (reduce_dim == 1) {
        // Reduce over dim1
        if (idx < batch_size * dim2) {
            int b = idx / dim2;
            int d = idx % dim2;
            
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            for (int i = 0; i < dim1; i++) {
                scalar_t val = input[b * dim1 * dim2 + i * dim2 + d];
                max_val = max(max_val, val);
            }
            output[idx] = max_val;
        }
    }
    else if (reduce_dim == 2) {
        // Reduce over dim2
        if (idx < batch_size * dim1) {
            int b = idx / dim1;
            int d = idx % dim1;
            
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            for (int i = 0; i < dim2; i++) {
                scalar_t val = input[b * dim1 * dim2 + d * dim2 + i];
                max_val = max(max_val, val);
            }
            output[idx] = max_val;
        }
    }
}

torch::Tensor max_reduce_cuda(torch::Tensor input, int reduce_dim) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int dim1 = sizes[1];
    int dim2 = sizes[2];
    
    torch::Tensor output;
    if (reduce_dim == 1) {
        output = torch::empty({batch_size, dim2}, input.options());
    } else {
        output = torch::empty({batch_size, dim1}, input.options());
    }

    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_reduce_cuda", ([&] {
        max_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, dim1, dim2, reduce_dim
        );
    }));

    return output;
}
"""

max_reduce_cpp_source = """
torch::Tensor max_reduce_cuda(torch::Tensor input, int reduce_dim);
"""

max_reduce = load_inline(
    name='max_reduce',
    cpp_sources=max_reduce_cpp_source,
    cuda_sources=max_reduce_source,
    functions=['max_reduce_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_reduce = max_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduce.max_reduce_cuda(x.cuda(), self.dim)