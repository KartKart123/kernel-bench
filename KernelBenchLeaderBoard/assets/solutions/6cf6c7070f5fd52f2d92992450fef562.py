import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus activation
softplus_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void softplus_kernel(const float* __restrict__ input, float* __restrict__ output, int size, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float y;
        if (x > threshold) {
            y = x;  // avoid overflow
        } else if (x < -threshold) {
            y = expf(x);  // safe to compute exp(x)
        } else if (x > 0.0f) {
            y = x + log1pf(expf(-x));
        } else {
            y = log1pf(expf(x));
        }
        output[idx] = y;
    }
}

torch::Tensor softplus_cuda(torch::Tensor input, float threshold) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be a float tensor");

    auto output = torch::empty_like(input);

    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    softplus_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, threshold);

    return output;
}
"""

softplus_cpp_source = """
torch::Tensor softplus_cuda(torch::Tensor input, float threshold);
"""

# Compile the inline CUDA code for Softplus activation
softplus_module = load_inline(
    name='softplus_cuda',
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_cuda_source,
    functions=['softplus_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus = softplus_module

    def forward(self, x):
        threshold = 20.0  # This is the default threshold in PyTorch
        return self.softplus.softplus_cuda(x, threshold)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed