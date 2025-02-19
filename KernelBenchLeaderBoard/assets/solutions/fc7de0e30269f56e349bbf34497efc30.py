import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code for sum reduction over a specified dimension
sum_reduce_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for sum reduction over dimension 1
__global__ void sum_reduce_dim1_kernel(const float* __restrict__ x, float* __restrict__ y, int N, int D1, int D2)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && k < D2)
    {
        float sum = 0.0f;
        for (int j = 0; j < D1; ++j)
        {
            sum += x[n * D1 * D2 + j * D2 + k];
        }
        y[n * D2 + k] = sum;
    }
}

// C++ interface
torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x)
{
    // Ensure the tensor is contiguous
    x = x.contiguous();

    auto N = x.size(0);
    auto D1 = x.size(1);
    auto D2 = x.size(2);

    // Allocate output tensor
    auto y = torch::zeros({N, 1, D2}, x.options());

    const int threads = 16;
    dim3 block_size(threads, threads);
    dim3 grid_size((N + threads - 1) / threads, (D2 + threads - 1) / threads);

    sum_reduce_dim1_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, D1, D2
    );

    return y;
}
"""

sum_reduce_cpp_source = """
torch::Tensor sum_reduce_dim1_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code for sum reduction
sum_reduce = load_inline(
    name="sum_reduce",
    cpp_sources=sum_reduce_cpp_source,
    cuda_sources=sum_reduce_source,
    functions=["sum_reduce_dim1_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        assert dim == 1, "Only reduction over dimension 1 is supported."
        self.dim = dim
        self.sum_reduce = sum_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_reduce.sum_reduce_dim1_cuda(x)