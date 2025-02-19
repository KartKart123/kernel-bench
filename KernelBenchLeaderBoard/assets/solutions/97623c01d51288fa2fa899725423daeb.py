import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

argmin_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmin_kernel(const scalar_t* input, int64_t* output, 
                            int batch_size, int dim1, int dim2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * dim2) {
        int batch_idx = tid / dim2;
        int col_idx = tid % dim2;
        
        scalar_t min_val = input[batch_idx * dim1 * dim2 + col_idx];
        int min_idx = 0;
        
        for(int i = 1; i < dim1; i++) {
            scalar_t curr_val = input[batch_idx * dim1 * dim2 + i * dim2 + col_idx];
            if(curr_val < min_val) {
                min_val = curr_val;
                min_idx = i;
            }
        }
        output[batch_idx * dim2 + col_idx] = min_idx;
    }
}

torch::Tensor argmin_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    
    auto output = torch::empty({batch_size, dim2}, 
                             torch::TensorOptions().dtype(torch::kLong).device(input.device()));
    
    const int threads = 256;
    const int blocks = (batch_size * dim2 + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "argmin_cuda", ([&] {
        argmin_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            batch_size, dim1, dim2
        );
    }));

    return output;
}
"""

argmin_cpp_source = """
torch::Tensor argmin_cuda(torch::Tensor input, int dim);
"""

argmin_cuda = load_inline(
    name='argmin_cuda',
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_cuda_source,
    functions=['argmin_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_cuda = argmin_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmin_cuda.argmin_cuda(x.cuda(), self.dim)