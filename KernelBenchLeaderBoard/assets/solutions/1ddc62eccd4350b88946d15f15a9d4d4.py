import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = (val + 3.0f) / 6.0f;
        val = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
        output[idx] = val;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    hardsigmoid_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

hardsigmoid_cpp_source = """
torch::Tensor hardsigmoid_cuda(torch::Tensor input);
"""

hardsigmoid_cuda = load_inline(
    name='hardsigmoid_cuda',
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=['hardsigmoid_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardsigmoid_cuda = hardsigmoid_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid_cuda.hardsigmoid_cuda(x.cuda())