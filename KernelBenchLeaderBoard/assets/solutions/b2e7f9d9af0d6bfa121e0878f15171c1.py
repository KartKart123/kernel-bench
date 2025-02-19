import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused tanh activation, scaling, and bias addition
fused_tanh_scale_bias_cpp_source = """
torch::Tensor fused_tanh_scale_bias_cuda(torch::Tensor x, float scaling_factor, torch::Tensor bias);
"""

fused_tanh_scale_bias_cuda_source = """
#include <torch/extension.h>

__global__ void fused_tanh_scale_bias_kernel(const float* x, float scaling_factor, const float* bias, float* out, int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    if (idx < total_elements) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (channels * height * width);

        int index = ((b * channels + c) * height + h) * width + w;

        float val = x[index];
        val = tanhf(val) * scaling_factor + bias[c];
        out[index] = val;
    }
}

torch::Tensor fused_tanh_scale_bias_cuda(torch::Tensor x, float scaling_factor, torch::Tensor bias) {
    auto x_cont = x.contiguous();
    auto bias_cont = bias.contiguous();

    int batch_size = x_cont.size(0);
    int channels = x_cont.size(1);
    int height = x_cont.size(2);
    int width = x_cont.size(3);

    auto out = torch::empty_like(x_cont);

    int total_elements = batch_size * channels * height * width;

    const int threads = 1024;
    const int blocks = (total_elements + threads -1) / threads;

    fused_tanh_scale_bias_kernel<<<blocks, threads>>>(x_cont.data_ptr<float>(), scaling_factor, bias_cont.data_ptr<float>(), out.data_ptr<float>(), batch_size, channels, height, width);

    // Return the output tensor
    return out;
}
"""

# Compile the inline CUDA code for fused tanh, scaling, and bias addition
fused_tanh_scale_bias = load_inline(
    name='fused_tanh_scale_bias',
    cpp_sources=fused_tanh_scale_bias_cpp_source,
    cuda_sources=fused_tanh_scale_bias_cuda_source,
    functions=['fused_tanh_scale_bias_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model using fused custom CUDA kernel for tanh activation, scaling, and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.fused_tanh_scale_bias = fused_tanh_scale_bias

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Fused tanh activation, scaling, and bias addition
        x = self.fused_tanh_scale_bias.fused_tanh_scale_bias_cuda(x, self.scaling_factor, self.bias.view(-1))
        # Max-pooling
        x = self.max_pool(x)
        return x