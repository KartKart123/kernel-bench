import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void instance_norm_kernel(const float* input, float* output, float* mean, float* var, int batch_size, int num_features, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_features * height * width;

    if (idx < total_size) {
        int b = (idx / (num_features * height * width)) % batch_size;
        int c = (idx / (height * width)) % num_features;
        int h = (idx / width) % height;
        int w = idx % width;

        float mean_val = mean[b * num_features + c];
        float var_val = var[b * num_features + c];
        float inv_std = rsqrtf(var_val + 1e-5);

        output[idx] = (input[idx] - mean_val) * inv_std;
    }
}

std::vector<torch::Tensor> instance_norm_cuda(torch::Tensor input) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto mean = torch::zeros({batch_size, num_features}, options);
    auto var = torch::zeros({batch_size, num_features}, options);
    auto output = torch::zeros_like(input);

    // Compute mean and variance
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_features; ++c) {
            auto input_slice = input.index({b, c}).flatten();
            mean.index({b, c}) = input_slice.mean();
            var.index({b, c}) = input_slice.var(false);
        }
    }

    const int block_size = 256;
    const int num_blocks = (batch_size * num_features * height * width + block_size - 1) / block_size;

    instance_norm_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), batch_size, num_features, height, width);

    return {output};
}
"""

instance_norm_cpp_source = "std::vector<torch::Tensor> instance_norm_cuda(torch::Tensor input);"

# Compile the inline CUDA code for Instance Normalization
instance_norm = load_inline(
    name='instance_norm',
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=['instance_norm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_cuda(x)[0]