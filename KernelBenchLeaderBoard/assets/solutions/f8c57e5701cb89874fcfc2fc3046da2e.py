import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Layer Normalization
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void layernorm_kernel(const float* x, float* out, float* mean, float* var, int size, int normalized_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            float val = x[idx * normalized_size + i];
            sum += val;
            sq_sum += val * val;
        }
        mean[idx] = sum / normalized_size;
        var[idx] = sq_sum / normalized_size - mean[idx] * mean[idx];

        float inv_std = rsqrtf(var[idx] + 1e-5f);
        for (int i = 0; i < normalized_size; ++i) {
            out[idx * normalized_size + i] = (x[idx * normalized_size + i] - mean[idx]) * inv_std;
        }
    }
}

std::vector<torch::Tensor> layernorm_cuda(torch::Tensor x, int normalized_size) {
    auto size = x.size(0);
    auto out = torch::zeros_like(x);
    auto mean = torch::zeros(size, x.options());
    auto var = torch::zeros(size, x.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    layernorm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), size, normalized_size);

    return {out};
}
"""

layernorm_cpp_source = "std::vector<torch::Tensor> layernorm_cuda(torch::Tensor x, int normalized_size);"

# Compile the inline CUDA code for Layer Normalization
layernorm = load_inline(
    name='layernorm',
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_source,
    functions=['layernorm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Layer Normalization using custom CUDA kernel.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(ModelNew, self).__init__()
        self.normalized_size = int(torch.prod(torch.tensor(normalized_shape)))
        self.layernorm = layernorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.layernorm.layernorm_cuda(x, self.normalized_size)[0]