import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

swish_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}

torch::Tensor swish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    swish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

swish_cpp_source = """
torch::Tensor swish_cuda(torch::Tensor input);
"""

swish_cuda = load_inline(
    name='swish_cuda',
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_cuda_source,
    functions=['swish_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish_cuda = swish_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish_cuda.swish_cuda(x.cuda())

def get_inputs():
    x = torch.randn(16, 16384).cuda()
    return [x]

def get_init_inputs():
    return []