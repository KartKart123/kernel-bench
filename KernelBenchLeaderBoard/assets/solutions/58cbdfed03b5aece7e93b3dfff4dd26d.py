import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel
fused_kernel_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* __restrict__ x, const float* __restrict__ wsum, float* __restrict__ y, float scaling_factor, int input_size) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    extern __shared__ float shared_data[];

    float sum = 0.0f;
    for (int i = thread_idx; i < input_size; i += blockDim.x) {
        sum += x[batch_idx * input_size + i] * wsum[i];
    }

    shared_data[thread_idx] = sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        y[batch_idx] = shared_data[0] * scaling_factor;
    }
}

torch::Tensor fused_kernel_launcher(torch::Tensor x, torch::Tensor wsum, float scaling_factor) {
    int batch_size = x.size(0);
    int input_size = x.size(1);

    auto y = torch::zeros({batch_size}, x.options());

    const int threads = 256;
    const int blocks = batch_size;
    const size_t shared_memory_size = threads * sizeof(float);

    fused_kernel<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(),
        wsum.data_ptr<float>(),
        y.data_ptr<float>(),
        scaling_factor,
        input_size
    );

    return y;
}
"""

fused_kernel_cpp_source = """
torch::Tensor fused_kernel_launcher(torch::Tensor x, torch::Tensor wsum, float scaling_factor);
"""

# Compile the custom CUDA kernel
fused_module = load_inline(
    name='fused_module',
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_code,
    functions=['fused_kernel_launcher'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs the same computation using a custom CUDA kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_kernel = fused_module

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Recompute wsum
        wsum = torch.sum(self.weight, dim=0).contiguous()
        wsum = wsum.to(x.device).float()

        scaling_factor = self.scaling_factor / 2.0

        x = x.contiguous().float()

        output = self.fused_kernel.fused_kernel_launcher(
            x, wsum, scaling_factor
        ).unsqueeze(1)
        return output