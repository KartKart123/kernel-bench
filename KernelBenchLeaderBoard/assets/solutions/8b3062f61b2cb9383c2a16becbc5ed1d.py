import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA code for the custom HardTanh activation
hardtanh_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        if (val > max_val)
            output[idx] = max_val;
        else if (val < min_val)
            output[idx] = min_val;
        else
            output[idx] = val;
    }
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    hardtanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, min_val, max_val);

    return output;
}
"""

hardtanh_cpp_source = "torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);"

# Compile the CUDA code
hardtanh = load_inline(
    name='custom_hardtanh',
    cpp_sources=hardtanh_cpp_source,
    cuda_sources=hardtanh_cuda_source,
    functions=['hardtanh_cuda'],
    verbose=True,
    extra_cuda_cflags=[]
)

class ModelNew(nn.Module):
    """
    Simple model that performs a HardTanh activation with custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hardtanh = hardtanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        # Ensure the input tensor is on the GPU
        if not x.is_cuda:
            x = x.cuda()
        # Assuming min_val=-1.0 and max_val=1.0 as in the original code
        return self.hardtanh.hardtanh_cuda(x, -1.0, 1.0)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed