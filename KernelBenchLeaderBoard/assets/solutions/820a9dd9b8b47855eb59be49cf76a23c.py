import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define CUDA source code
custom_kernel_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_activation_kernel(const float* __restrict__ x, float* __restrict__ y, float add_value, float scale, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x_i = x[idx];

        // Compute mish activation
        // mish(x) = x * tanh(ln(1 + exp(x)))
        float sp;
        if (x_i > 20.0f) {
            sp = x_i;
        } else if (x_i < -20.0f) {
            sp = expf(x_i);
        } else {
            sp = log1pf(expf(x_i));
        }

        float mish = x_i * tanhf(sp);

        mish = mish + add_value;
        mish = fminf(fmaxf(mish, -1.0f), 1.0f);
        y[idx] = mish * scale;
    }
}

torch::Tensor custom_activation_cuda(torch::Tensor x, float add_value, float scale) {
    auto y = torch::empty_like(x);

    int num_elements = x.numel();

    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    custom_activation_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), add_value, scale, num_elements);

    return y;
}
'''

custom_kernel_cpp_source = '''
torch::Tensor custom_activation_cuda(torch::Tensor x, float add_value, float scale);
'''

custom_kernel = load_inline(
    name='custom_kernel',
    cpp_sources=custom_kernel_cpp_source,
    cuda_sources=custom_kernel_source,
    functions=['custom_activation_cuda'],
    verbose=False,
    extra_cuda_cflags=['-D_USE_MATH_DEFINES']
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs ConvTranspose2d followed by fused custom activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.custom_activation = custom_kernel.custom_activation_cuda
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.custom_activation(x, self.add_value, self.scale)
        return x

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]