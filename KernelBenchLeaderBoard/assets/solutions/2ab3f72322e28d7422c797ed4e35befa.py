import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    tanh_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

tanh_cpp_source = "torch::Tensor tanh_cuda(torch::Tensor input);"

custom_tanh = load_inline(
    name='custom_tanh',
    cpp_sources=tanh_cpp_source,
    cuda_sources=tanh_source,
    functions=['tanh_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_tanh = custom_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_tanh.tanh_cuda(x.cuda())

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []