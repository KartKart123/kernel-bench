import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int batch_size, int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / seq_len;
    int seq_idx = tid % seq_len;
    
    if (batch_idx < batch_size && seq_idx < seq_len) {
        float sum = 0.0f;
        int offset = batch_idx * seq_len;
        
        for (int i = 0; i <= seq_idx; i++) {
            if (mask[offset + i]) {
                sum += x[offset + i];
            }
        }
        out[offset + seq_idx] = sum;
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask) {
    auto batch_size = x.size(0);
    auto seq_len = x.size(1);
    auto out = torch::zeros_like(x);
    
    const int threads_per_block = 256;
    const int total_elements = batch_size * seq_len;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    masked_cumsum_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        batch_size,
        seq_len
    );
    
    return out;
}
"""

masked_cumsum_cpp_source = """
torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask);
"""

masked_cumsum = load_inline(
    name='masked_cumsum',
    cpp_sources=masked_cumsum_cpp_source,
    cuda_sources=masked_cumsum_source,
    functions=['masked_cumsum_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.masked_cumsum = masked_cumsum

    def forward(self, x, mask):
        if x.is_cuda and mask.is_cuda:
            return self.masked_cumsum.masked_cumsum_cuda(x, mask)
        return torch.cumsum(x * mask, dim=self.dim)