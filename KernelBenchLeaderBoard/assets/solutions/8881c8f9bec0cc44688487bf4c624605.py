import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute per-row sum of absolute values
__global__ void sum_abs_kernel(const float* __restrict__ x, float* __restrict__ sum_abs, int N, int D) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D;
    while (index < total_elements) {
        int n = index / D;
        float val = fabsf(x[index]);
        atomicAdd(&sum_abs[n], val);
        index += blockDim.x * gridDim.x;
    }
}

// Kernel to normalize x[n, d] = x[n, d] / sum_abs[n]
__global__ void normalize_kernel(float* x, const float* __restrict__ sum_abs, int N, int D) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D;
    while (index < total_elements) {
        int n = index / D;
        x[index] = x[index] / sum_abs[n];
        index += blockDim.x * gridDim.x;
    }
}

torch::Tensor l1_normalize(torch::Tensor x) {
    // x: input tensor of shape (N, D)
    int N = x.size(0);
    int D = x.size(1);

    auto sum_abs = torch::zeros({N}, x.options());

    int threads = 256;
    int blocks = min((N * D + threads - 1) / threads, 1024);

    sum_abs_kernel<<<blocks, threads>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), N, D);
    cudaDeviceSynchronize();
    normalize_kernel<<<blocks, threads>>>(x.data_ptr<float>(), sum_abs.data_ptr<float>(), N, D);

    return x;
}

"""

l1_norm_cpp_source = """
torch::Tensor l1_normalize(torch::Tensor x);
"""

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name='l1_norm',
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=['l1_normalize'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x):
        x = x.contiguous()
        return self.l1_norm.l1_normalize(x)