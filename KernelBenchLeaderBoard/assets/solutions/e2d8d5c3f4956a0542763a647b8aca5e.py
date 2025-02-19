import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_reduction_kernel(const float* input, float* output, 
                                    int reduce_dim_size, int outer_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outer_size * inner_size) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;
        
        float sum = 0.0f;
        for (int i = 0; i < reduce_dim_size; i++) {
            int input_idx = outer_idx * reduce_dim_size * inner_size + 
                           i * inner_size + inner_idx;
            sum += input[input_idx];
        }
        output[idx] = sum / reduce_dim_size;
    }
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    auto input_sizes = input.sizes();
    int ndim = input_sizes.size();
    dim = dim < 0 ? dim + ndim : dim;
    
    int reduce_dim_size = input_sizes[dim];
    
    int outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input_sizes[i];
    }
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= input_sizes[i];
    }
    
    std::vector<int64_t> output_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_sizes.push_back(input_sizes[i]);
        }
    }
    
    auto output = torch::empty(output_sizes, input.options());
    
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    
    mean_reduction_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        reduce_dim_size,
        outer_size,
        inner_size
    );
    
    return output;
}
"""

mean_reduction_cpp_source = """
torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);
"""

mean_reduction = load_inline(
    name='mean_reduction',
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=['mean_reduction_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduction.mean_reduction_cuda(x.cuda(), self.dim)