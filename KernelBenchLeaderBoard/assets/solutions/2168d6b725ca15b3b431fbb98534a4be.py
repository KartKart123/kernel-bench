import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the inline CUDA code for the fused operation
fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float gelu(float x) {
    // Approximate GELU activation
    // Use the approximation: x * 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    const float c = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845608f;
    float x_cube = x * x * x;
    return x * 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + c * x_cube)));
}

__global__ void fused_kernel(const float *x, float *y, int batch_size, int out_features, int kernel_size, float scale_factor, int L_out) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const float *x_batch = x + batch_idx * out_features;

    float max_value = -FLT_MAX;
    for (int window_idx = 0; window_idx < L_out; ++window_idx) {
        int start_idx = window_idx * kernel_size;
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            sum += x_batch[start_idx + i];
        }
        float avg = sum / kernel_size;
        float gelu_avg = gelu(avg);
        float scaled_value = gelu_avg * scale_factor;
        if (scaled_value > max_value) {
            max_value = scaled_value;
        }
    }
    y[batch_idx] = max_value;
}

torch::Tensor fused_forward(torch::Tensor x, int kernel_size, float scale_factor) {
    int batch_size = x.size(0);
    int out_features = x.size(1);
    int L_out = out_features / kernel_size;  // Assume that out_features is divisible by kernel_size

    auto y = torch::empty({batch_size}, x.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, out_features, kernel_size, scale_factor, L_out);

    return y;
}
"""

fused_cpp_source = "torch::Tensor fused_forward(torch::Tensor x, int kernel_size, float scale_factor);"

# Compile the inline CUDA code for the fused operation
fused_module = load_inline(
    name='fused_module',
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=['fused_forward'],
    verbose=True,
    extra_cflags=[''],
    extra_cuda_cflags=['-O3'],
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA kernels for fused operations.
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.fused_func = fused_module.fused_forward

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_func(x, self.kernel_size, self.scale_factor)
        return x

def get_inputs():
    return [torch.randn(128, 512).cuda()]

def get_init_inputs():
    return [512, 256, 4, 2.0]