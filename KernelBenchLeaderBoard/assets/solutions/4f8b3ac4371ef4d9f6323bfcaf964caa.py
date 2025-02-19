import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softsign_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softsign_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float abs_x = abs(x);
        output[idx] = x / (1.0f + abs_x);
    }
}

torch::Tensor softsign_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    softsign_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

softsign_cpp_source = """
torch::Tensor softsign_cuda(torch::Tensor input);
"""

softsign_cuda = load_inline(
    name='softsign_cuda',
    cpp_sources=softsign_cpp_source,
    cuda_sources=softsign_source,
    functions=['softsign_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softsign_cuda = softsign_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softsign_cuda.softsign_cuda(x.cuda())

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []