import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = batch_size * channels * height * width;
    
    if (idx < size) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (width * height * channels);
        
        float x = input[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float scale = gamma[c];
        float shift = beta[c];
        
        output[idx] = scale * (x - mean) / sqrt(var + epsilon) + shift;
    }
}

std::vector<torch::Tensor> batch_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
) {
    auto output = torch::zeros_like(input);
    
    int batch_size = input.size(0);
    int channels = input.size(1); 
    int height = input.size(2);
    int width = input.size(3);
    
    int size = batch_size * channels * height * width;
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    batch_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels, 
        height,
        width,
        epsilon
    );
    
    return {output};
}
"""

batch_norm_cpp_source = """
std::vector<torch::Tensor> batch_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta, 
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float epsilon
);
"""

batch_norm_cuda = load_inline(
    name='batch_norm',
    cpp_sources=batch_norm_cpp_source,
    cuda_sources=batch_norm_source,
    functions=['batch_norm_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.batch_norm = batch_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm.batch_norm_cuda(
            x.cuda(),
            self.weight.cuda(),
            self.bias.cuda(),
            self.running_mean.cuda(),
            self.running_var.cuda(),
            self.eps
        )[0]