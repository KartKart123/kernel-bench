import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Frobenius norm normalization
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void frobenius_norm_kernel(const float* x, float* out, float* norm, int size) {
    __shared__ float shared_sum[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    if (idx < size) {
        val = x[idx] * x[idx];
    }

    shared_sum[threadIdx.x] = val;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Accumulate block results
    if (threadIdx.x == 0) {
        atomicAdd(norm, shared_sum[0]);
    }
}

__global__ void normalize_kernel(const float* x, float* out, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    auto norm = torch::zeros({1}, x.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    frobenius_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), norm.data_ptr<float>(), size);
    cudaDeviceSynchronize();

    float h_norm;
    cudaMemcpy(&h_norm, norm.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    h_norm = sqrt(h_norm);

    normalize_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), h_norm, size);

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
    """
    Optimized model that performs Frobenius norm normalization using a custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return self.frobenius_norm.frobenius_norm_cuda(x)