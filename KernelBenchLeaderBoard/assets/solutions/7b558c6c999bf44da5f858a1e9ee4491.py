import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matmul, scaling, and residual addition
fused_matmul_scale_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_scale_add_kernel(const float* input, const float* weight, const float* bias, float* output, float scaling_factor, int in_features, int out_features, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int batch = idx / out_features;
        int feature = idx % out_features;
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[batch * in_features + i] * weight[feature * in_features + i];
        }
        sum += bias[feature];
        output[idx] = sum * scaling_factor + sum;
    }
}

torch::Tensor fused_matmul_scale_add_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor, int in_features, int out_features, int batch_size) {
    auto output = torch::zeros({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

    fused_matmul_scale_add_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        in_features,
        out_features,
        batch_size
    );

    return output;
}
"""

fused_matmul_scale_add_cpp_source = "torch::Tensor fused_matmul_scale_add_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float scaling_factor, int in_features, int out_features, int batch_size);"

# Compile the inline CUDA code for fused matmul, scaling, and residual addition
fused_matmul_scale_add = load_inline(
    name='fused_matmul_scale_add',
    cpp_sources=fused_matmul_scale_add_cpp_source,
    cuda_sources=fused_matmul_scale_add_source,
    functions=['fused_matmul_scale_add_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.fused_matmul_scale_add = fused_matmul_scale_add

    def forward(self, x):
        weight = self.matmul.weight.t()
        bias = self.matmul.bias
        return self.fused_matmul_scale_add.fused_matmul_scale_add_cuda(x, weight, bias, self.scaling_factor, self.matmul.in_features, self.matmul.out_features, x.size(0))