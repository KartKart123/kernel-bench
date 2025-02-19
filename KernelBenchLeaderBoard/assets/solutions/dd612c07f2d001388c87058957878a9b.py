import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

scan_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scan_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = tid / seq_len;
    int seq_id = tid % seq_len;
    
    if (batch_id >= batch_size) return;
    
    float sum = 0;
    for (int i = 0; i <= seq_id; i++) {
        sum += input[batch_id * seq_len + i];
    }
    output[tid] = sum;
}

torch::Tensor scan_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto output = torch::zeros_like(input);
    
    const int threads_per_block = 256;
    const int total_elements = batch_size * seq_len;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    scan_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len
    );
    
    return output;
}
"""

scan_cpp_source = """
torch::Tensor scan_cuda(torch::Tensor input);
"""

scan_op = load_inline(
    name='scan_cuda',
    cpp_sources=scan_cpp_source,
    cuda_sources=scan_cuda_source,
    functions=['scan_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.scan = scan_op

    def forward(self, x):
        return self.scan.scan_cuda(x)