import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Batch Normalization
batchnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batchnorm_kernel(const float* x, float* out, const float* mean, const float* var, const float* gamma, const float* beta, int batch_size, int num_features, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_features * dim1 * dim2;

    if (idx < total_size) {
        int b = (idx / (num_features * dim1 * dim2)) % batch_size;
        int f = (idx / (dim1 * dim2)) % num_features;
        int d1 = (idx / dim2) % dim1;
        int d2 = idx % dim2;

        float x_norm = (x[idx] - mean[f]) / sqrtf(var[f] + 1e-5);
        out[idx] = gamma[f] * x_norm + beta[f];
    }
}

torch::Tensor batchnorm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta) {
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    auto dim1 = x.size(2);
    auto dim2 = x.size(3);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * num_features * dim1 * dim2 + block_size - 1) / block_size;

    batchnorm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), batch_size, num_features, dim1, dim2);

    return out;
}
"""

batchnorm_cpp_source = "torch::Tensor batchnorm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta);"

# Compile the inline CUDA code for Batch Normalization
batchnorm = load_inline(
    name='batchnorm',
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_source,
    functions=['batchnorm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.batchnorm = batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var

        return self.batchnorm.batchnorm_cuda(x, mean.cuda(), var.cuda(), self.gamma.cuda(), self.beta.cuda())