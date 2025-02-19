import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_reduction_kernel(const float* input, float* output, 
                                   int batch_size, int dim1, int dim2, int reduce_dim) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int dim_size;
    int outer_size;
    int inner_size;
    
    if (reduce_dim == 1) {
        dim_size = dim1;
        outer_size = batch_size;
        inner_size = dim2;
    } else if (reduce_dim == 0) {
        dim_size = batch_size;
        outer_size = 1;
        inner_size = dim1 * dim2;
    } else {
        dim_size = dim2;
        outer_size = batch_size * dim1;
        inner_size = 1;
    }
    
    int outer_idx = bid / inner_size;
    int inner_idx = bid % inner_size;
    
    // Load input into shared memory
    float sum = 0.0f;
    for(int i = tid; i < dim_size; i += blockDim.x) {
        int idx;
        if (reduce_dim == 1) {
            idx = outer_idx * dim1 * dim2 + i * dim2 + inner_idx;
        } else if (reduce_dim == 0) {
            idx = i * dim1 * dim2 + inner_idx;
        } else {
            idx = outer_idx * dim2 + i;
        }
        sum += input[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if(tid == 0) {
        output[bid] = sdata[0];
    }
}

torch::Tensor sum_reduction_cuda(torch::Tensor input, int reduce_dim) {
    const int batch_size = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);
    
    std::vector<int64_t> output_shape;
    int num_blocks;
    
    if (reduce_dim == 1) {
        output_shape = {batch_size, 1, dim2};
        num_blocks = batch_size * dim2;
    } else if (reduce_dim == 0) {
        output_shape = {1, dim1, dim2};
        num_blocks = dim1 * dim2;
    } else {
        output_shape = {batch_size, dim1, 1};
        num_blocks = batch_size * dim1;
    }
    
    auto output = torch::zeros(output_shape, input.options());
    
    const int block_size = 256;
    const int shared_mem_size = block_size * sizeof(float);
    
    sum_reduction_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, dim1, dim2, reduce_dim
    );
    
    return output;
}
"""

sum_reduction_cpp_source = """
torch::Tensor sum_reduction_cuda(torch::Tensor input, int reduce_dim);
"""

sum_reduction = load_inline(
    name='sum_reduction',
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=['sum_reduction_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.sum_reduction = sum_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduction.sum_reduction_cuda(x.cuda(), self.dim)