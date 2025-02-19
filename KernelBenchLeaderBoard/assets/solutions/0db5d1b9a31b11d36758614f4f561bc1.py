import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void frobenius_norm_kernel(const float* x, float* norm, int size) {
    __shared__ float cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cache_idx = threadIdx.x;

    float sum = 0.0f;
    while (idx < size) {
        sum += x[idx] * x[idx];
        idx += blockDim.x * gridDim.x;
    }

    cache[cache_idx] = sum;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (cache_idx < i) {
            cache[cache_idx] += cache[cache_idx + i];
        }
        __syncthreads();
    }

    if (cache_idx == 0) {
        atomicAdd(norm, cache[0]);
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto norm = torch::zeros(1, torch::TensorOptions().device(x.device()).dtype(x.dtype()));

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    frobenius_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), norm.data_ptr<float>(), size);

    return x / norm.sqrt();
}
"""

frobenius_norm_cpp_source = (
    "torch::Tensor frobenius_norm_cuda(torch::Tensor x);"
)

frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_norm.frobenius_norm_cuda(x)