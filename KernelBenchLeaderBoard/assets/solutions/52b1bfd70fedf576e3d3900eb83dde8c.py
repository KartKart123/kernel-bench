import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA code

cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_reduce_dim1_kernel(const float* x, float* out, int N, int D1, int D2)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int d2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (n < N && d2 < D2)
    {
        float min_val = x[n * D1 * D2 + 0 * D2 + d2];
        for (int d1 = 1; d1 < D1; ++d1)
        {
            float val = x[n * D1 * D2 + d1 * D2 + d2];
            if (val < min_val)
                min_val = val;
        }
        out[n * D2 + d2] = min_val;
    }
}

torch::Tensor min_reduce_dim1_cuda(torch::Tensor x)
{
    // x is of shape [N, D1, D2]
    int N = x.size(0);
    int D1 = x.size(1);
    int D2 = x.size(2);

    auto out = torch::empty({N, D2}, x.options());

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (D2 + threads - 1) / threads);

    min_reduce_dim1_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        D1,
        D2);

    return out;
}
'''

cpp_source = '''
torch::Tensor min_reduce_dim1_cuda(torch::Tensor x);
'''

# Compile the inline CUDA code
min_reduce_dim1 = load_inline(
    name='min_reduce_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['min_reduce_dim1_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over dimension 1 using custom CUDA kernels.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over. Only supports dim=1 in this implementation.
        """
        super(ModelNew, self).__init__()
        assert dim == 1, "This implementation only supports reduction over dimension 1."
        self.dim = dim
        self.min_reduce_dim1 = min_reduce_dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over dimension 1 to the input tensor using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape [N, D1, D2].

        Returns:
            torch.Tensor: Output tensor after min reduction of shape [N, D2].
        """
        x = x.contiguous()
        return self.min_reduce_dim1.min_reduce_dim1_cuda(x)