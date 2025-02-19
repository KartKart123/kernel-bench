import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* x, float* out, int batch_size, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_features) {
        int batch_idx = idx / num_features;
        int feature_idx = idx % num_features;

        float max_val = -INFINITY;
        for (int i = 0; i < num_features; i++) {
            if (x[batch_idx * num_features + i] > max_val) {
                max_val = x[batch_idx * num_features + i];
            }
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < num_features; i++) {
            sum_exp += expf(x[batch_idx * num_features + i] - max_val);
        }

        out[idx] = expf(x[idx] - max_val) / sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * num_features + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, num_features);

    return out;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax.softmax_cuda(x)