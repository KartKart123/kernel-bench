import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

min_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void min_reduction_kernel(const float* input, float* output, 
                                   int batch_size, int dim1, int dim2, int reduce_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (reduce_dim == 1) {
        // Reduce over dim1
        if (idx < batch_size * dim2) {
            int batch_idx = idx / dim2;
            int d2_idx = idx % dim2;
            
            float min_val = std::numeric_limits<float>::max();
            for (int d1 = 0; d1 < dim1; d1++) {
                float val = input[batch_idx * dim1 * dim2 + d1 * dim2 + d2_idx];
                min_val = min(min_val, val);
            }
            output[idx] = min_val;
        }
    }
    else if (reduce_dim == 2) {
        // Reduce over dim2
        if (idx < batch_size * dim1) {
            int batch_idx = idx / dim1;
            int d1_idx = idx % dim1;
            
            float min_val = std::numeric_limits<float>::max();
            for (int d2 = 0; d2 < dim2; d2++) {
                float val = input[batch_idx * dim1 * dim2 + d1_idx * dim2 + d2];
                min_val = min(min_val, val);
            }
            output[idx] = min_val;
        }
    }
}

torch::Tensor min_reduction_cuda(torch::Tensor input, int reduce_dim) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    
    torch::Tensor output;
    if (reduce_dim == 1) {
        output = torch::empty({batch_size, dim2}, input.options());
    } else {
        output = torch::empty({batch_size, dim1}, input.options());
    }
    
    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;
    
    min_reduction_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, dim1, dim2, reduce_dim
    );
    
    return output;
}
"""

min_reduction_cpp_source = """
torch::Tensor min_reduction_cuda(torch::Tensor input, int reduce_dim);
"""

min_reduction = load_inline(
    name='min_reduction',
    cpp_sources=min_reduction_cpp_source,
    cuda_sources=min_reduction_source,
    functions=['min_reduction_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduction = min_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x.cuda(), self.dim)