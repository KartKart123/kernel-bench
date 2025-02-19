import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void log_softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    
    // Compute max for numerical stability
    float max_val = -INFINITY;
    for(int i = 0; i < dim; i++) {
        float val = input[batch_idx * dim + i];
        max_val = max(max_val, val);
    }
    __syncthreads();
    
    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for(int i = 0; i < dim; i++) {
        sum += exp(input[batch_idx * dim + i] - max_val);
    }
    float log_sum = log(sum);
    __syncthreads();
    
    // Compute final output
    for(int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[batch_idx * dim + i] = input[batch_idx * dim + i] - max_val - log_sum;
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    auto output = torch::empty_like(input);
    const int batch_size = input.size(0);
    const int feature_dim = input.size(1);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    log_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        feature_dim
    );
    
    return output;
}
"""

log_softmax_cpp_source = """
torch::Tensor log_softmax_cuda(torch::Tensor input, int dim);
"""

log_softmax_cuda = load_inline(
    name='log_softmax_cuda',
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=['log_softmax_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax_cuda = log_softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_softmax_cuda.log_softmax_cuda(x.cuda(), self.dim)