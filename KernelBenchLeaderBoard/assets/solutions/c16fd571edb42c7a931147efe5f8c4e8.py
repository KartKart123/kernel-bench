import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Layer Normalization
layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layer_norm_kernel(const float* input, const float* weight, const float* bias, float* output, 
                                   const float* mean, const float* inv_var, int batch_size, int features, int dim1, int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * features * dim1 * dim2) {
        int feature_idx = (idx / (dim1 * dim2)) % features;
        output[idx] = (input[idx] - mean[feature_idx]) * inv_var[feature_idx] * weight[feature_idx] + bias[feature_idx];
    }
}

__global__ void compute_mean_kernel(const float* input, float* mean, int batch_size, int features, int dim1, int dim2) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < features) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size * dim1 * dim2; i++) {
            sum += input[feature_idx * batch_size * dim1 * dim2 + i];
        }
        mean[feature_idx] = sum / (batch_size * dim1 * dim2);
    }
}

__global__ void compute_inv_var_kernel(const float* input, const float* mean, float* inv_var, int batch_size, int features, int dim1, int dim2) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature_idx < features) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size * dim1 * dim2; i++) {
            float diff = input[feature_idx * batch_size * dim1 * dim2 + i] - mean[feature_idx];
            sum += diff * diff;
        }
        inv_var[feature_idx] = 1.0f / sqrt(sum / (batch_size * dim1 * dim2) + 1e-5f);
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto dim1 = input.size(2);
    auto dim2 = input.size(3);
    auto output = torch::zeros_like(input);
    auto mean = torch::zeros(features, input.device());
    auto inv_var = torch::zeros(features, input.device());

    const int block_size = 256;
    const int num_blocks = (features + block_size - 1) / block_size;

    compute_mean_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), batch_size, features, dim1, dim2);
    compute_inv_var_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), inv_var.data_ptr<float>(), batch_size, features, dim1, dim2);

    const int output_block_size = 256;
    const int output_num_blocks = (batch_size * features * dim1 * dim2 + output_block_size - 1) / output_block_size;

    layer_norm_kernel<<<output_num_blocks, output_block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), inv_var.data_ptr<float>(), batch_size, features, dim1, dim2);

    return output;
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code for Layer Normalization
layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.layer_norm = layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm.layer_norm_cuda(x, self.weight, self.bias)