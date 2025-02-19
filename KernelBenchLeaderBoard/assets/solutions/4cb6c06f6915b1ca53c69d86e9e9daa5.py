import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ out, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_data[];

    // Load data into shared memory
    float max_val = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, x[batch_idx * dim + i]);
    }

    shared_data[tid] = max_val;
    __syncthreads();

    // Reduce to find the maximum value
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    max_val = shared_data[0];

    // Compute the exponentials and sum them
    float sum_exp = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        out[batch_idx * dim + i] = expf(x[batch_idx * dim + i] - max_val);
        sum_exp += out[batch_idx * dim + i];
    }

    shared_data[tid] = sum_exp;
    __syncthreads();

    // Reduce to find the sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    sum_exp = shared_data[0];

    // Normalize to get the softmax
    for (int i = tid; i < dim; i += blockDim.x) {
        out[batch_idx * dim + i] /= sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int shared_mem_size = block_size * sizeof(float);

    softmax_kernel<<<batch_size, block_size, shared_mem_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name='softmax',
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=['softmax_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return self.softmax.softmax_cuda(x)