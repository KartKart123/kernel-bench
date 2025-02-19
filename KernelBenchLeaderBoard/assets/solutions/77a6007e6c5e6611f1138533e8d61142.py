import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min along channels and tanh(tanh) activation
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void min_tanh2_kernel(const float* x, float* out, int batch_size, int channels, int height, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int total_elements = batch_size * height * width;

    if (index < total_elements) {
        int b = index / (height * width);
        int idx_in_batch = index % (height * width);
        int h = idx_in_batch / width;
        int w = idx_in_batch % width;

        // Compute min over channels
        float min_val = x[((b * channels + 0) * height + h) * width + w];
        for (int c = 1; c < channels; ++c) {
            float val = x[((b * channels + c) * height + h) * width + w];
            if (val < min_val) {
                min_val = val;
            }
        }

        // Apply tanh twice
        float y = tanhf(tanhf(min_val));

        // Store the result
        out[((b * height + h) * width) + w] = y;
    }
}

torch::Tensor min_tanh2_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    auto out = torch::zeros({batch_size, 1, height, width}, x.options());

    int total_elements = batch_size * height * width;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    min_tanh2_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, channels, height, width);

    return out;
}
"""

cpp_source = """
torch::Tensor min_tanh2_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code for the custom operation
min_tanh2 = load_inline(
    name='min_tanh2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['min_tanh2_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.min_tanh2 = min_tanh2

    def forward(self, x):
        x = self.conv(x)
        x = self.min_tanh2.min_tanh2_cuda(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]