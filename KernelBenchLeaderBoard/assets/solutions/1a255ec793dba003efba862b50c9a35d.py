import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for reverse cumulative sum along axis 1 for 2D tensors
reverse_cumsum_cuda_src = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(const float *x, float *y, int N, int M) {
    int row = blockIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int j = M - 1; j >= 0; --j) {
            int idx = row * M + j;
            sum += x[idx];
            y[idx] = sum;
        }
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor x) {
    auto N = x.size(0);
    auto M = x.size(1);
    auto y = torch::zeros_like(x);

    const int threads = 1;  // one thread per block
    const int blocks = N;

    reverse_cumsum_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), N, M);

    // Wait for the CUDA kernel to finish
    cudaDeviceSynchronize();

    return y;
}
"""

reverse_cumsum_cpp_sources = """
torch::Tensor reverse_cumsum_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
reverse_cumsum = load_inline(
    name='reverse_cumsum',
    cpp_sources=reverse_cumsum_cpp_sources,
    cuda_sources=reverse_cumsum_cuda_src,
    functions=['reverse_cumsum_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        assert dim == 1, "This implementation only supports dim=1"
        self.dim = dim
        self.reverse_cumsum = reverse_cumsum

    def forward(self, x):
        # x should be a 2D tensor
        assert x.dim() == 2, "Input tensor must be 2D"
        return self.reverse_cumsum.reverse_cumsum_cuda(x)