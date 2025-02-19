import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void softplus_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Softplus implementation: log(1 + exp(x))
        // Using numerically stable version to avoid overflow
        const float threshold = 20.0f;
        float x = input[idx];
        if (x > threshold) {
            output[idx] = x;  // For large x, softplus(x) ≈ x
        } else {
            output[idx] = logf(1.0f + expf(x));
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    softplus_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

softplus_cpp_source = """
torch::Tensor softplus_cuda(torch::Tensor input);
"""

softplus_cuda = load_inline(
    name='softplus_cuda',
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=['softplus_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus_cuda = softplus_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus_cuda.softplus_cuda(x.cuda())