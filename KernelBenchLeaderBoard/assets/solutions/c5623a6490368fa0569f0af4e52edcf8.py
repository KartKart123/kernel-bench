import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Frobenius norm normalization
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void frobenius_norm_kernel(const float* x, float* norm, int size) {
    __shared__ float shared_mem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;

    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        local_sum += x[i] * x[i];
    }

    shared_mem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(norm, shared_mem[0]);
    }
}

__global__ void elementwise_div_kernel(const float* x, float norm, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto norm = torch::zeros(1, x.options()).cuda();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    frobenius_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), norm.data_ptr<float>(), size);

    norm = torch::sqrt(norm);

    auto out = torch::zeros_like(x);
    elementwise_div_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), norm.item<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

frobenius_norm_cpp_source = "torch::Tensor frobenius_norm_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Frobenius norm normalization
frobenius_norm = load_inline(
    name='frobenius_norm',
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=['frobenius_norm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_norm.frobenius_norm_cuda(x)