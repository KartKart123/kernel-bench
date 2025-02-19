import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L1 normalization
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l1_norm_kernel(const float* x, float* out, int dim, int batch_size) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    // Calculate the L1 norm for the current batch
    float l1_norm = 0.0;
    for (int i = thread_idx; i < dim; i += blockDim.x) {
        l1_norm += fabsf(x[batch_idx * dim + i]);
    }

    // Use shared memory to accumulate the L1 norm
    __shared__ float shared_l1_norm[256];
    shared_l1_norm[thread_idx] = l1_norm;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_l1_norm[thread_idx] += shared_l1_norm[thread_idx + stride];
        }
        __syncthreads();
    }

    // Normalize the input tensor
    l1_norm = shared_l1_norm[0];
    for (int i = thread_idx; i < dim; i += blockDim.x) {
        out[batch_idx * dim + i] = x[batch_idx * dim + i] / l1_norm;
    }
}

torch::Tensor l1_norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = batch_size;

    l1_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), dim, batch_size);

    return out;
}
"""

l1_norm_cpp_source = "torch::Tensor l1_norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code for L1 normalization
l1_norm = load_inline(
    name='l1_norm',
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=['l1_norm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using a custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()
        self.l1_norm = l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return self.l1_norm.l1_norm_cuda(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []