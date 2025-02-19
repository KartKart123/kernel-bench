import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda(torch::Tensor input);"

# Compile the inline CUDA code for ReLU activation
relu_module = load_inline(
    name='relu_cuda',
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_cuda_source,
    functions=['relu_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a ReLU activation using custom CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu_cuda = relu_module.relu_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom CUDA ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        # Ensure the input is on CUDA device
        if not x.is_cuda:
            x = x.cuda()
        return self.relu_cuda(x)
    
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed