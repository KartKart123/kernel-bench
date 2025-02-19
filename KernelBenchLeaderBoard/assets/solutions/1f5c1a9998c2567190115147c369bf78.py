import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardTanh
hardtanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* x, float* out, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val < min_val) {
            out[idx] = min_val;
        } else if (val > max_val) {
            out[idx] = max_val;
        } else {
            out[idx] = val;
        }
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor x, float min_val, float max_val) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    hardtanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, min_val, max_val);

    return out;
}
"""

hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor x, float min_val, float max_val);"

# Compile the inline CUDA code for HardTanh
hardtanh = load_inline(
    name='hardtanh',
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_source,
    functions=['hardtanh_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardtanh = hardtanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardtanh.hardtanh_cuda(x, -1., 1.)