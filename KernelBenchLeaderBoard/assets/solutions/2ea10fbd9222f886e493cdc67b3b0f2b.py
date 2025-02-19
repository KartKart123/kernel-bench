import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void l2_norm_kernel(const float* x, float* out, int dim, int stride) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    // Calculate the L2 norm for the given batch
    float norm = 0.0;
    for (int i = thread_idx; i < dim; i += blockDim.x) {
        float val = x[batch_idx * stride + i];
        norm += val * val;
    }

    // Reduce within the block
    __shared__ float shared_norm[256];
    shared_norm[thread_idx] = norm;
    __syncthreads();

    // Perform reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (thread_idx < offset) {
            shared_norm[thread_idx] += shared_norm[thread_idx + offset];
        }
        __syncthreads();
    }

    // Normalize the input
    if (thread_idx == 0) {
        norm = sqrtf(shared_norm[0]);
        for (int i = 0; i < dim; ++i) {
            out[batch_idx * stride + i] = x[batch_idx * stride + i] / norm;
        }
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto dim = x.size(1);
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = batch_size;

    l2_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), dim, dim);

    return out;
}
"""

l2_norm_cpp_source = "torch::Tensor l2_norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code for L2 normalization
l2_norm = load_inline(
    name='l2_norm',
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=['l2_norm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.l2_norm = l2_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_norm.l2_norm_cuda(x)