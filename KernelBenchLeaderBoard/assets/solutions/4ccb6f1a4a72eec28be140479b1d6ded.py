import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumprod_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumprod_kernel(const float* input, float* output, int batch_size, int dim_size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get starting index for this batch
        int batch_offset = batch_idx * dim_size;
        
        // First element is copied as-is
        if (tid == 0) {
            output[batch_offset] = input[batch_offset];
        }
        __syncthreads();
        
        // Compute cumulative product
        for (int i = 1; i < dim_size; i++) {
            if (tid == i) {
                output[batch_offset + i] = output[batch_offset + i - 1] * input[batch_offset + i];
            }
            __syncthreads();
        }
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto dim_size = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const int threads_per_block = 1024;
    const int blocks = batch_size;
    
    cumprod_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim_size
    );
    
    return output;
}
"""

cumprod_cpp_source = """
torch::Tensor cumprod_cuda(torch::Tensor input, int dim);
"""

cumprod_cuda = load_inline(
    name='cumprod_cuda',
    cpp_sources=cumprod_cpp_source,
    cuda_sources=cumprod_cuda_source,
    functions=['cumprod_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cumprod_cuda = cumprod_cuda

    def forward(self, x):
        return self.cumprod_cuda.cumprod_cuda(x, self.dim)