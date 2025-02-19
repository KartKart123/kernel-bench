import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

product_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void product_reduction_kernel(const float* input, float* output, 
                                       int batch_size, int dim1, int dim2, int reduction_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (reduction_dim == 1) {
        // Reduce over dim1
        int batch_idx = tid / dim2;
        int d2_idx = tid % dim2;
        
        if (batch_idx < batch_size && d2_idx < dim2) {
            float prod = 1.0f;
            for (int d1_idx = 0; d1_idx < dim1; d1_idx++) {
                prod *= input[batch_idx * dim1 * dim2 + d1_idx * dim2 + d2_idx];
            }
            output[batch_idx * dim2 + d2_idx] = prod;
        }
    }
}

torch::Tensor product_reduction_cuda(torch::Tensor input, int reduction_dim) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    
    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    
    torch::Tensor output;
    if (reduction_dim == 1) {
        output = torch::empty({batch_size, dim2}, options);
    }
    
    const int threads = 256;
    const int blocks = (batch_size * dim2 + threads - 1) / threads;
    
    product_reduction_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, dim1, dim2, reduction_dim
    );
    
    return output;
}
"""

product_reduction_cpp_source = """
torch::Tensor product_reduction_cuda(torch::Tensor input, int reduction_dim);
"""

product_reduction = load_inline(
    name='product_reduction',
    cpp_sources=product_reduction_cpp_source,
    cuda_sources=product_reduction_source,
    functions=['product_reduction_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.product_reduction = product_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.product_reduction.product_reduction_cuda(x.cuda(), self.dim)