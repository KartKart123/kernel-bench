import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean reduction
mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_reduction_kernel(const float* input, float* output, int dim1, int dim2, int dim3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < dim1 * dim3; i += stride) {
        int d1 = i / dim3;
        int d3 = i % dim3;
        float sum = 0.0f;
        for (int d2 = 0; d2 < dim2; ++d2) {
            sum += input[d1 * dim2 * dim3 + d2 * dim3 + d3];
        }
        output[d1 * dim3 + d3] = sum / dim2;
    }
}

torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim) {
    auto dim1 = input.size(0);
    auto dim2 = input.size(1);
    auto dim3 = input.size(2);
    auto output = torch::zeros({dim1, dim3}, input.options());

    const int block_size = 256;
    const int num_blocks = (dim1 * dim3 + block_size - 1) / block_size;

    mean_reduction_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), dim1, dim2, dim3);

    return output;
}
"""

mean_reduction_cpp_source = "torch::Tensor mean_reduction_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code for mean reduction
mean_reduction = load_inline(
    name='mean_reduction',
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=['mean_reduction_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_reduction.mean_reduction_cuda(x, self.dim)