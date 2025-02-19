import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_divide_cpp_source = """
torch::Tensor elementwise_divide_cuda(torch::Tensor input, float divisor);
"""

elementwise_divide_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_divide_kernel(const float* input, float* output, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / divisor;
    }
}

torch::Tensor elementwise_divide_cuda(torch::Tensor input, float divisor) {
    auto output = torch::empty_like(input);
    int size = input.numel();

    const int threads = 512;
    const int blocks = (size + threads - 1) / threads;

    elementwise_divide_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), divisor, size);

    return output;
}
"""

elementwise_divide = load_inline(
    name='elementwise_divide',
    cpp_sources=elementwise_divide_cpp_source,
    cuda_sources=elementwise_divide_cuda_source,
    functions=['elementwise_divide_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.elementwise_divide = elementwise_divide
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = self.elementwise_divide.elementwise_divide_cuda(x, self.divisor)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x