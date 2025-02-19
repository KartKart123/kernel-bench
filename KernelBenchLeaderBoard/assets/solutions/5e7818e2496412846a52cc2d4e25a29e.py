import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmin along dim=1
argmin_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void argmin_kernel(const float* x, int64_t* output_indices, int batch_size, int dim1, int dim2)
{
    int batch_idx = blockIdx.x;
    int dim2_idx = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float* svals = sdata;
    int* sidx = (int*)&svals[blockDim.x];

    float val = FLT_MAX;
    int idx = -1;

    if (tid < dim1)
    {
        int index = batch_idx * dim1 * dim2 + tid * dim2 + dim2_idx;
        val = x[index];
        idx = tid;
    }

    svals[tid] = val;
    sidx[tid] = idx;
    __syncthreads();

    // Reduction to find min value and index
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (svals[tid + s] < svals[tid])
            {
                svals[tid] = svals[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output_indices[batch_idx * dim2 + dim2_idx] = sidx[0];
    }
}

torch::Tensor argmin_cuda(torch::Tensor x)
{
    int batch_size = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
    auto output_indices = torch::empty({batch_size, dim2}, options);

    dim3 grid(batch_size, dim2);
    dim3 block(dim1);

    int shared_mem_size = sizeof(float) * dim1 + sizeof(int) * dim1;

    argmin_kernel<<<grid, block, shared_mem_size>>>(x.data_ptr<float>(), output_indices.data_ptr<int64_t>(), batch_size, dim1, dim2);

    return output_indices;
}
"""

argmin_cpp_source = """
torch::Tensor argmin_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code for argmin
argmin = load_inline(
    name='argmin',
    cpp_sources=argmin_cpp_source,
    cuda_sources=argmin_cuda_source,
    functions=['argmin_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmin_cuda = argmin.argmin_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        if self.dim == 1 and x.is_cuda:
            return self.argmin_cuda(x)
        else:
            # For other dimensions or if tensor is not on CUDA, fallback to torch.argmin
            return torch.argmin(x, dim=self.dim)

batch_size = 16
dim1 = 256
dim2 = 256
dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [dim]