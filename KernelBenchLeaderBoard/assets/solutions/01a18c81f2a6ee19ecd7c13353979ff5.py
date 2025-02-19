import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void rms_norm_kernel(const float* __restrict__ x, float* __restrict__ out,
                                int batch_size, int num_features, int dim1, int dim2, float eps) {
    // Each block corresponds to one (b, i, j)
    // Threads correspond to c
    int idx = blockIdx.x; // block index
    int c = threadIdx.x;  // thread index

    if (idx >= batch_size * dim1 * dim2 || c >= num_features)
        return;

    int j = idx % dim2;
    int temp = idx / dim2;
    int i = temp % dim1;
    int b = temp / dim1;

    // Compute the index for x and out
    int x_idx = ((b * num_features + c) * dim1 + i) * dim2 + j;

    // Each thread computes x[b, c, i, j]^2
    float val = x[x_idx];
    float val_sq = val * val;

    // Use shared memory for reduction
    extern __shared__ float sdata[];

    sdata[c] = val_sq;
    __syncthreads();

    // Perform reduction to compute sum
    // Assuming num_features is power of 2
    for (unsigned int s = num_features / 2; s > 0; s >>= 1) {
        if (c < s) {
            sdata[c] += sdata[c + s];
        }
        __syncthreads();
    }

    float rms = 0.0f;
    if (c == 0) {
        float mean = sdata[0] / num_features;
        rms = sqrtf(mean + eps);
        sdata[0] = rms;
    }
    __syncthreads();

    rms = sdata[0];

    // Each thread normalizes its own x[b, c, i, j]
    out[x_idx] = val / rms;
}

torch::Tensor rms_norm_cuda(torch::Tensor x, float eps) {
    const auto batch_size = x.size(0);
    const auto num_features = x.size(1);
    const auto dim1 = x.size(2);
    const auto dim2 = x.size(3);

    auto out = torch::empty_like(x);

    const int threads_per_block = num_features;
    const int blocks = batch_size * dim1 * dim2;

    const int shared_mem_size = threads_per_block * sizeof(float);

    rms_norm_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        num_features,
        dim1,
        dim2,
        eps
    );

    return out;
}
"""

rms_norm_cpp_source = """
torch::Tensor rms_norm_cuda(torch::Tensor x, float eps);
"""

# Compile the CUDA extension
rms_norm = load_inline(
    name='rms_norm',
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_cuda_source,
    functions=['rms_norm_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized model that performs RMS Normalization using custom CUDA kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm = rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm.rms_norm_cuda(x, self.eps)