import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        size
    );
    
    return output;
}
"""

leaky_relu_cpp_source = """
torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope);
"""

leaky_relu_cuda = load_inline(
    name='leaky_relu_cuda',
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=['leaky_relu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leaky_relu_cuda = leaky_relu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu_cuda.leaky_relu_cuda(x.cuda(), self.negative_slope)