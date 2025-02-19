import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int64_t* output, 
                            int batch_size, int dim1, int dim2, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dim == 1) {
        if (idx < batch_size * dim2) {
            int batch = idx / dim2;
            int d2 = idx % dim2;
            
            scalar_t max_val = input[batch * dim1 * dim2 + d2];
            int64_t max_idx = 0;
            
            for (int d1 = 0; d1 < dim1; d1++) {
                scalar_t val = input[batch * dim1 * dim2 + d1 * dim2 + d2];
                if (val > max_val) {
                    max_val = val;
                    max_idx = d1;
                }
            }
            output[idx] = max_idx;
        }
    }
    else if (dim == 2) {
        if (idx < batch_size * dim1) {
            int batch = idx / dim1;
            int d1 = idx % dim1;
            
            scalar_t max_val = input[batch * dim1 * dim2 + d1 * dim2];
            int64_t max_idx = 0;
            
            for (int d2 = 0; d2 < dim2; d2++) {
                scalar_t val = input[batch * dim1 * dim2 + d1 * dim2 + d2];
                if (val > max_val) {
                    max_val = val;
                    max_idx = d2;
                }
            }
            output[idx] = max_idx;
        }
    }
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int dim1 = sizes[1];
    int dim2 = sizes[2];
    
    torch::Tensor output;
    if (dim == 1) {
        output = torch::empty({batch_size, dim2}, torch::kLong).to(input.device());
    } else {
        output = torch::empty({batch_size, dim1}, torch::kLong).to(input.device());
    }
    
    const int threads = 256;
    const int blocks = (dim == 1) ? 
        (batch_size * dim2 + threads - 1) / threads :
        (batch_size * dim1 + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            batch_size, dim1, dim2, dim
        );
    }));
    
    return output;
}
"""

argmax_cpp_source = """
torch::Tensor argmax_cuda(torch::Tensor input, int dim);
"""

argmax_cuda = load_inline(
    name='argmax_cuda',
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_cuda_source,
    functions=['argmax_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_cuda = argmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax_cuda.argmax_cuda(x.cuda(), self.dim)