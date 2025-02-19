import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel source code
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* x_in, float* x_out, float scaling_factor, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float x = x_in[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        x_out[idx] = x + scaling_factor * sig;
    }
}

torch::Tensor fused_forward(torch::Tensor x_in, float scaling_factor) {
    int size = x_in.numel();
    auto x_out = torch::empty_like(x_in);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(x_in.data_ptr<float>(), x_out.data_ptr<float>(), scaling_factor, size);

    return x_out;
}
"""

cpp_source = """
torch::Tensor fused_forward(torch::Tensor x_in, float scaling_factor);
"""

# Compile the CUDA extension
fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_forward'],
    extra_cuda_cflags=['--expt-relaxed-constexpr'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd" with custom CUDA kernels.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.gemm(x)
        x = self.fused_ops.fused_forward(x, self.scaling_factor)
        return x