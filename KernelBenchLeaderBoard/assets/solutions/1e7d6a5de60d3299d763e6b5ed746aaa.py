import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / seq_len;
    int seq_idx = tid % seq_len;
    
    if (batch_idx < batch_size && seq_idx < seq_len) {
        float sum = 0;
        for (int i = seq_idx; i < seq_len; i++) {
            sum += input[batch_idx * seq_len + i];
        }
        output[batch_idx * seq_len + seq_idx] = sum;
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto output = torch::zeros_like(input);
    
    const int threads_per_block = 256;
    const int total_elements = batch_size * seq_len;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    reverse_cumsum_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len
    );
    
    return output;
}
"""

reverse_cumsum_cpp_source = """
torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim);
"""

reverse_cumsum = load_inline(
    name='reverse_cumsum',
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=['reverse_cumsum_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)