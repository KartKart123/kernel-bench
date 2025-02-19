import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for L2 normalization
l2_normalize_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void l2_normalize_kernel(const float* x, float* y, int batch_size, int dim) {
    int sample_idx = blockIdx.x;
    const float* x_sample = x + sample_idx * dim;
    float* y_sample = y + sample_idx * dim;

    extern __shared__ float sdata[];

    // Each thread computes partial sum of squares
    float partial_sum = 0.0f;
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        float val = x_sample[idx];
        partial_sum += val * val;
    }

    // Each thread writes partial sum to shared memory
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // After reduction, sdata[0] contains the total sum
    float norm = sqrtf(sdata[0] + 1e-8f); // Adding small epsilon to prevent division by zero

    // Now, each thread normalizes its elements
    for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
        y_sample[idx] = x_sample[idx] / norm;
    }
}

torch::Tensor l2_normalize_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);
    auto y = torch::empty_like(x);

    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem_size = threads * sizeof(float);

    l2_normalize_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), batch_size, dim
    );

    return y;
}
"""

l2_normalize_cpp_source = """
torch::Tensor l2_normalize_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code for L2 normalization
l2_normalize = load_inline(
    name='l2_normalize',
    cpp_sources=l2_normalize_cpp_source,
    cuda_sources=l2_normalize_source,
    functions=['l2_normalize_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return l2_normalize.l2_normalize_cuda(x)