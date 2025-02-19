import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l2_norm_kernel(const float* input, float* output, const int batch_size, const int dim) {
    const int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Calculate L2 norm for this batch element
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = input[batch_idx * dim + i];
        norm += val * val;
    }
    norm = sqrtf(norm);

    // Normalize the values
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[batch_idx * dim + i] = input[batch_idx * dim + i] / norm;
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    l2_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
    
    return output;
}
"""

l2_norm_cpp_source = """
torch::Tensor l2_norm_cuda(torch::Tensor input);
"""

l2_norm = load_inline(
    name='l2_norm',
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=['l2_norm_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_norm.l2_norm_cuda(x.cuda())

def get_inputs():
    x = torch.randn(16, 16384).cuda()
    return [x]

def get_init_inputs():
    return []