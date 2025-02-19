import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_over_dim1_cuda = """
#include <torch/extension.h>

__global__ void mean_over_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int dim1, int dim2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * dim2;

    if (index < total_elements) {
        int b = index / dim2;
        int d2 = index % dim2;

        float sum = 0.0f;
        for (int d1 = 0; d1 < dim1; d1++) {
            sum += input[(b * dim1 + d1) * dim2 + d2];
        }
        output[b * dim2 + d2] = sum / dim1;
    }
}

torch::Tensor mean_over_dim1_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);

    auto output = torch::zeros({batch_size, dim2}, input.options());

    int total_elements = batch_size * dim2;
    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    mean_over_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim1,
        dim2
    );

    return output;
}
"""

mean_over_dim1_cpp = """
torch::Tensor mean_over_dim1_cuda(torch::Tensor input);
"""

mean_over_dim1 = load_inline(
    name='mean_over_dim1',
    cpp_sources=mean_over_dim1_cpp,
    cuda_sources=mean_over_dim1_cuda,
    functions=['mean_over_dim1_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_over_dim1 = mean_over_dim1.mean_over_dim1_cuda

    def forward(self, x):
        if x.is_cuda and self.dim == 1 and x.dtype == torch.float32:
            return self.mean_over_dim1(x)
        else:
            return torch.mean(x, dim=self.dim)