import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the fused operation
fused_operation_source = """

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_kernel(const float* __restrict__ x_in, float* __restrict__ x_out,
                             const float add_value, const float multiply_value, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = x_in[idx] + add_value;
        temp = fminf(temp, 0.0f);
        temp = gelu_approx(temp);
        x_out[idx] = temp * multiply_value;
    }
}

torch::Tensor fused_operation(torch::Tensor x_in, float add_value, float multiply_value) {
    auto x_out = torch::empty_like(x_in);
    int size = x_in.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    fused_kernel<<<blocks, threads>>>(x_in.data_ptr<float>(), x_out.data_ptr<float>(),
                                      add_value, multiply_value, size);
    return x_out;
}
"""

fused_operation_cpp_source = """
torch::Tensor fused_operation(torch::Tensor x_in, float add_value, float multiply_value);
"""

# Compile the inline CUDA code for the fused operation
fused_operation = load_inline(
    name='fused_operation',
    cpp_sources=fused_operation_cpp_source,
    cuda_sources=fused_operation_source,
    functions=['fused_operation'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized Model that uses a fused custom CUDA kernel for element-wise operations after the transposed convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.fused_operation = fused_operation

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_operation.fused_operation(x, self.add_value, self.multiply_value)
        return x

batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]