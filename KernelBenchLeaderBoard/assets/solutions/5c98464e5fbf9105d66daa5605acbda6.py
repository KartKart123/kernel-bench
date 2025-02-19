import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

elu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elu_kernel(const float* input, float* output, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0 ? x : alpha * (exp(x) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor input, float alpha) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        alpha,
        size
    );

    return output;
}
"""

elu_cpp_source = """
torch::Tensor elu_cuda(torch::Tensor input, float alpha);
"""

elu_cuda = load_inline(
    name='elu_cuda',
    cpp_sources=elu_cpp_source,
    cuda_sources=elu_cuda_source,
    functions=['elu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elu_cuda = elu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu_cuda.elu_cuda(x.cuda(), self.alpha)