import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val < min_val ? min_val : (val > max_val ? max_val : val);
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    hardtanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        min_val,
        max_val
    );
    
    return output;
}
"""

hardtanh_cpp_source = """
torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);
"""

hardtanh_cuda = load_inline(
    name='hardtanh_cuda',
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=['hardtanh_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardtanh_cuda = hardtanh_cuda
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh_cuda.hardtanh_cuda(x.cuda(), -1.0, 1.0)