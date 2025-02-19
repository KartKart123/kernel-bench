import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instancenorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instancenorm_kernel(const float* input, float* output, const float* mean, const float* invvar, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels * height * width) {
        int b = idx / (channels * height * width);
        int c = (idx / (height * width)) % channels;
        int h = (idx / width) % height;
        int w = idx % width;

        float x = input[idx];
        float m = mean[b * channels + c];
        float iv = invvar[b * channels + c];

        output[idx] = (x - m) * iv;
    }
}

__global__ void compute_mean_kernel(const float* input, float* mean, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        int b = idx / channels;
        int c = idx % channels;

        float sum = 0.0f;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                sum += input[b * channels * height * width + c * height * width + h * width + w];
            }
        }

        mean[idx] = sum / (height * width);
    }
}

__global__ void compute_invvar_kernel(const float* input, const float* mean, float* invvar, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * channels) {
        int b = idx / channels;
        int c = idx % channels;

        float sum = 0.0f;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float x = input[b * channels * height * width + c * height * width + h * width + w];
                float m = mean[b * channels + c];
                sum += (x - m) * (x - m);
            }
        }

        invvar[idx] = 1.0f / sqrt(sum / (height * width) + 1e-5);
    }
}

torch::Tensor instancenorm_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto mean = torch::zeros({batch_size, channels}, torch::TensorOptions().device(input.device()));
    auto invvar = torch::zeros({batch_size, channels}, torch::TensorOptions().device(input.device()));
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks_mean = (batch_size * channels + block_size - 1) / block_size;
    const int num_blocks_invvar = (batch_size * channels + block_size - 1) / block_size;
    const int num_blocks_norm = (batch_size * channels * height * width + block_size - 1) / block_size;

    compute_mean_kernel<<<num_blocks_mean, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), batch_size, channels, height, width);
    compute_invvar_kernel<<<num_blocks_invvar, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), invvar.data_ptr<float>(), batch_size, channels, height, width);
    instancenorm_kernel<<<num_blocks_norm, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), mean.data_ptr<float>(), invvar.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}
"""

instancenorm_cpp_source = (
    "torch::Tensor instancenorm_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Instance Normalization
instancenorm = load_inline(
    name="instancenorm",
    cpp_sources=instancenorm_cpp_source,
    cuda_sources=instancenorm_source,
    functions=["instancenorm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.instancenorm = instancenorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instancenorm.instancenorm_cuda(x)