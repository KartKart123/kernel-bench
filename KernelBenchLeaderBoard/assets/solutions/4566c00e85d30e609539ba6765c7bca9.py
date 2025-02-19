import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for computing mish(mish(x))
mish_mish_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline float mish(float x) {
    float sp = log1pf(expf(x));  // softplus(x) = log(1 + exp(x))
    return x * tanhf(sp);
}

__global__ void mish_mish_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float mish_x = mish(x);
        output[idx] = mish(mish_x);
    }
}

torch::Tensor mish_mish_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);

    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    mish_mish_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

mish_mish_cpp_source = """
#include <torch/extension.h>

torch::Tensor mish_mish_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code
mish_mish = load_inline(
    name='mish_mish',
    cpp_sources=mish_mish_cpp_source,
    cuda_sources=mish_mish_cuda_source,
    functions=['mish_mish_cuda'],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.mish_mish = mish_mish

    def forward(self, x):
        x = self.conv(x)
        x = self.mish_mish.mish_mish_cuda(x)
        return x