import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMM + BatchNorm + GELU
fused_gemm_bn_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_gemm_bn_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int batch_idx = idx / out_features;
        int feature_idx = idx % out_features;

        // GEMM
        float sum = 0.0;
        for (int i = 0; i < in_features; ++i) {
            sum += input[batch_idx * in_features + i] * weight[feature_idx * in_features + i];
        }
        sum += bias[feature_idx];

        // BatchNorm
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float normed = (sum - mean) / sqrtf(var + eps);

        // GELU
        float gelu = 0.5 * normed * (1.0 + tanhf(0.7978845608 * (normed + 0.044715 * normed * normed * normed)));

        output[idx] = gelu;
    }
}

torch::Tensor fused_gemm_bn_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {

    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    auto output = torch::empty({batch_size, out_features}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;

    fused_gemm_bn_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps);

    return output;
}
"""

fused_gemm_bn_gelu_cpp_source = """
torch::Tensor fused_gemm_bn_gelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);
"""

# Compile the inline CUDA code for fused GEMM + BatchNorm + GELU
fused_gemm_bn_gelu = load_inline(
    name='fused_gemm_bn_gelu',
    cpp_sources=fused_gemm_bn_gelu_cpp_source,
    cuda_sources=fused_gemm_bn_gelu_source,
    functions=['fused_gemm_bn_gelu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.running_mean = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(out_features), requires_grad=False)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.eps = 1e-5

    def forward(self, x):
        x = fused_gemm_bn_gelu.fused_gemm_bn_gelu_cuda(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.eps)
        x = self.group_norm(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.relu(x)
        return x