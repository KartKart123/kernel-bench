import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean reduction
mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_reduction_kernel(const float* x, float* out, int batch_size, int dim1, int dim2, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim == 0) {
        if (idx < dim1 * dim2) {
            int i = idx / dim2;
            int j = idx % dim2;
            float sum = 0.0f;
            for (int k = 0; k < batch_size; k++) {
                sum += x[k * dim1 * dim2 + i * dim2 + j];
            }
            out[i * dim2 + j] = sum / batch_size;
        }
    } else if (dim == 1) {
        if (idx < batch_size * dim2) {
            int i = idx / dim2;
            int j = idx % dim2;
            float sum = 0.0f;
            for (int k = 0; k < dim1; k++) {
                sum += x[i * dim1 * dim2 + k * dim2 + j];
            }
            out[i * dim2 + j] = sum / dim1;
        }
    } else if (dim == 2) {
        if (idx < batch_size * dim1) {
            int i = idx / dim1;
            int j = idx % dim1;
            float sum = 0.0f;
            for (int k = 0; k < dim2; k++) {
                sum += x[i * dim1 * dim2 + j * dim2 + k];
            }
            out[i * dim1 + j] = sum / dim2;
        }
    }
}

torch::Tensor mean_reduction_cuda(torch::Tensor x, int dim) {
    int batch_size = x.size(0);
    int dim1 = x.size(1);
    int dim2 = x.size(2);
    torch::Tensor out;
    if (dim == 0) {
        out = torch::zeros({dim1, dim2}, torch::TensorOptions().device(x.device()));
    } else if (dim == 1) {
        out = torch::zeros({batch_size, dim2}, torch::TensorOptions().device(x.device()));
    } else if (dim == 2) {
        out = torch::zeros({batch_size, dim1}, torch::TensorOptions().device(x.device()));
    }

    const int block_size = 256;
    int num_blocks;
    if (dim == 0) {
        num_blocks = (dim1 * dim2 + block_size - 1) / block_size;
    } else if (dim == 1) {
        num_blocks = (batch_size * dim2 + block_size - 1) / block_size;
    } else if (dim == 2) {
        num_blocks = (batch_size * dim1 + block_size - 1) / block_size;
    }

    mean_reduction_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim1, dim2, dim);

    return out;
}
"""

mean_reduction_cpp_source = (
    "torch::Tensor mean_reduction_cuda(torch::Tensor x, int dim);"
)

# Compile the inline CUDA code for mean reduction
mean_reduction = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)