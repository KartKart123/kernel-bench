import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

torch::Tensor sigmoid_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

sigmoid_cpp_source = "torch::Tensor sigmoid_cuda(torch::Tensor input);"

custom_sigmoid = load_inline(
    name='custom_sigmoid',
    cpp_sources=sigmoid_cpp_source,
    cuda_sources=sigmoid_source,
    functions=['sigmoid_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_sigmoid = custom_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_sigmoid.sigmoid_cuda(x.cuda())

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []