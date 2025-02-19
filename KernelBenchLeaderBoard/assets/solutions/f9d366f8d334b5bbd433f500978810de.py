import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

selu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void selu_kernel(float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float alpha = 1.6732632423543772848170429916717;
        float scale = 1.0507009873554804934193349852946;
        float val = x[idx];
        out[idx] = scale * (val > 0 ? val : alpha * (exp(val) - 1));
    }
}

torch::Tensor selu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    selu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out;
}
"""

selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor x);"

selu_cuda = load_inline(
    name='selu_cuda',
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=['selu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda.selu_cuda(x.cuda())