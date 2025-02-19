import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Frobenius norm normalization
frobenius_norm_normalization_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void frobenius_norm_normalization_kernel(const float* x, float* out, float norm, int batch_size, int features, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features * dim1 * dim2) {
        out[idx] = x[idx] / norm;
    }
}

__global__ void frobenius_norm_kernel(const float* x, float* norm, int batch_size, int features, int dim1, int dim2) {
    __shared__ float cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    if (idx < batch_size * features * dim1 * dim2) {
        sum = x[idx] * x[idx];
    }

    cache[tid] = sum;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (tid < i) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm, cache[0]);
    }
}

torch::Tensor frobenius_norm_normalization_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto features = x.size(1);
    auto dim1 = x.size(2);
    auto dim2 = x.size(3);

    auto norm = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * features * dim1 * dim2 + block_size - 1) / block_size;

    frobenius_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), norm.data_ptr<float>(), batch_size, features, dim1, dim2);
    float norm_val = norm.item<float>();
    norm_val = sqrt(norm_val);

    frobenius_norm_normalization_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), norm_val, batch_size, features, dim1, dim2);

    return out;
}
"""

frobenius_norm_normalization_cpp_source = "torch::Tensor frobenius_norm_normalization_cuda(torch::Tensor x);"

# Compile the inline CUDA code for Frobenius norm normalization
frobenius_norm_normalization = load_inline(
    name='frobenius_norm_normalization',
    cpp_sources=frobenius_norm_normalization_cpp_source,
    cuda_sources=frobenius_norm_normalization_source,
    functions=['frobenius_norm_normalization_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_norm_normalization = frobenius_norm_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_norm_normalization.frobenius_norm_normalization_cuda(x)