import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

selu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void selu_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float xi = x[idx];
        const float alpha = 1.6732632423543772848170429916717f;
        const float scale = 1.0507009873554804934193349852946f;
        float result = scale * (xi > 0 ? xi : alpha * (expf(xi) - 1.0f));
        y[idx] = result;
    }
}

torch::Tensor selu_forward_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int N = x.numel();

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    selu_forward_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);

    return y;
}
"""

selu_cpp_source = """
torch::Tensor selu_forward_cuda(torch::Tensor x);
"""

selu_module = load_inline(
    name='selu_module',
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=['selu_forward_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_forward_cuda = selu_module.selu_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_forward_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed