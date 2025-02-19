import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Batch Normalization
batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_kernel(const float* input, float* output, const float* mean, const float* var, 
                                   const float* weight, const float* bias, float epsilon, int batch_size, int num_features, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_features * spatial_size) {
        int feature_idx = (idx / spatial_size) % num_features;
        float x = input[idx];
        float m = mean[feature_idx];
        float v = var[feature_idx];
        float w = weight[feature_idx];
        float b = bias[feature_idx];
        output[idx] = (x - m) / sqrt(v + epsilon) * w + b;
    }
}

torch::Tensor batch_norm_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, 
                              torch::Tensor weight, torch::Tensor bias, float epsilon) {
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto spatial_size = input.size(2) * input.size(3);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * num_features * spatial_size + block_size - 1) / block_size;

    batch_norm_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), 
                                                  mean.data_ptr<float>(), var.data_ptr<float>(), 
                                                  weight.data_ptr<float>(), bias.data_ptr<float>(), epsilon, 
                                                  batch_size, num_features, spatial_size);

    return output;
}
"""

batch_norm_cpp_source = (
    "torch::Tensor batch_norm_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, "
    "torch::Tensor weight, torch::Tensor bias, float epsilon);"
)

# Compile the inline CUDA code for Batch Normalization
batch_norm = load_inline(
    name="batch_norm",
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=["batch_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.mean = nn.Parameter(torch.zeros(num_features))
        self.var = nn.Parameter(torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.epsilon = 1e-5
        self.batch_norm = batch_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm.batch_norm_cuda(x, self.mean, self.var, self.weight, self.bias, self.epsilon)