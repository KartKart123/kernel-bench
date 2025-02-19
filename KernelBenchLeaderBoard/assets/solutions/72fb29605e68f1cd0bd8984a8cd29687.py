import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused tanh and subtraction operations
tanh_subtract_cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_subtract_kernel(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     const float subtract1_value,
                                     const float subtract2_value,
                                     const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = x[idx];
        val = val - subtract1_value;
        val = tanhf(val);
        val = val - subtract2_value;
        out[idx] = val;
    }
}

torch::Tensor tanh_subtract_cuda(torch::Tensor x, float subtract1_value, float subtract2_value) {
    const auto N = x.numel();
    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    tanh_subtract_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        subtract1_value,
        subtract2_value,
        N
    );

    return out;
}
'''

tanh_subtract_cuda_cpp = '''
torch::Tensor tanh_subtract_cuda(torch::Tensor x, float subtract1_value, float subtract2_value);
'''

# Compile the inline CUDA code for the fused operation
tanh_subtract = load_inline(
    name='tanh_subtract',
    cpp_sources=tanh_subtract_cuda_cpp,
    cuda_sources=tanh_subtract_cuda_source,
    functions=['tanh_subtract_cuda'],
    verbose=False,
    extra_cflags=[],
    extra_cuda_cflags=[],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model with custom fused CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.tanh_subtract = tanh_subtract

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh_subtract.tanh_subtract_cuda(x, self.subtract1_value, self.subtract2_value)
        x = self.avgpool(x)
        return x