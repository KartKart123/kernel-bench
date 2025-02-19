import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

relu_cuda = load_inline(
    name='relu_cuda',
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=['relu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_cuda = relu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_cuda.relu_cuda(x.cuda())

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []