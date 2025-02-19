import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float mish(float x) {
    return x * tanh(log1p(exp(x)));
}

__device__ float hardtanh(float x, float min_val, float max_val) {
    return fmaxf(fminf(x, max_val), min_val);
}

__global__ void fused_activation_kernel(float* input, float* output, int size, 
                                      float add_value, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        x = mish(x);
        x = x + add_value;
        x = hardtanh(x, -1.0f, 1.0f);
        x = x * scale;
        output[idx] = x;
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor input, float add_value, float scale) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    fused_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        add_value,
        scale
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_activation_cuda(torch::Tensor input, float add_value, float scale);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_activation_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_ops.fused_activation_cuda(x, self.add_value, self.scale)