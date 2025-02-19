import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA code for the fused operation
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

template <typename scalar_t>
__global__ void fused_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    float scaling_factor
    )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = batch_size * height * width;

    if (index < num_elements) {
        int n = index / (height * width);
        int h = (index / width) % height;
        int w = index % width;

        // Compute max over channels at (n, h, w)
        float max_val = -FLT_MAX;
        for (int c = 0; c < channels; ++c) {
            int idx = n * channels * height * width + c * height * width + h * width + w;
            max_val = fmaxf(max_val, x[idx]);
        }

        // Compute sum of exp(x - max_val) over channels at (n, h, w)
        float sum_exp = 0.0f;
        for (int c = 0; c < channels; ++c) {
            int idx = n * channels * height * width + c * height * width + h * width + w;
            sum_exp += expf(x[idx] - max_val);
        }

        // Compute softmax and the rest of the operations
        for (int c = 0; c < channels; ++c) {
            int idx = n * channels * height * width + c * height * width + h * width + w;
            float softmax_val = expf(x[idx] - max_val) / sum_exp;

            // Add bias
            float biased_val = softmax_val + bias[c];

            // Scale
            float scaled_val = biased_val * scaling_factor;

            // Apply sigmoid
            float sigmoid_val = 1.0f / (1.0f + expf(-scaled_val));

            // Write output
            output[idx] = sigmoid_val;
        }
    }
}

torch::Tensor fused_cuda_forward(
    torch::Tensor x,
    torch::Tensor bias,
    float scaling_factor
    )
{
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);

    auto output = torch::empty_like(x);

    int threads = 256;
    int blocks = (batch_size * height * width + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_cuda_forward", ([&] {
        fused_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            height,
            width,
            scaling_factor
            );
    }));

    return output;
}
"""

fused_kernel_header = """
torch::Tensor fused_cuda_forward(
    torch::Tensor x,
    torch::Tensor bias,
    float scaling_factor
    );
"""

# Compile the fused kernel
fused_kernel = load_inline(
    name='fused_kernel',
    cpp_sources=fused_kernel_header,
    cuda_sources=fused_kernel_source,
    functions=['fused_cuda_forward'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_kernel = fused_kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_kernel.fused_cuda_forward(x, self.bias.view(-1), self.scaling_factor)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]