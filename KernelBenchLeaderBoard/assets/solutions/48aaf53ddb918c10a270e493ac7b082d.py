import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSwish and ReLU fusion
hardswish_relu_fusion_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardswish_relu_fusion_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float hardswish = x * fmaxf(0.0f, fminf(6.0f, (x + 3.0f))) / 6.0f;
        output[idx] = fmaxf(0.0f, hardswish);
    }
}

torch::Tensor hardswish_relu_fusion_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardswish_relu_fusion_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

hardswish_relu_fusion_cpp_source = "torch::Tensor hardswish_relu_fusion_cuda(torch::Tensor input);"

# Compile the inline CUDA code for HardSwish and ReLU fusion
hardswish_relu_fusion = load_inline(
    name='hardswish_relu_fusion',
    cpp_sources=hardswish_relu_fusion_cpp_source,
    cuda_sources=hardswish_relu_fusion_source,
    functions=['hardswish_relu_fusion_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.hardswish_relu_fusion = hardswish_relu_fusion

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = self.hardswish_relu_fusion.hardswish_relu_fusion_cuda(x)
        return x