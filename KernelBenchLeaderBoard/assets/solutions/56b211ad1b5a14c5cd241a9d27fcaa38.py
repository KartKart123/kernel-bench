import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA code for argmax over dim=1 for 3D tensors
argmax_dim1_cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void argmax_dim1_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2
    ) {
    int n = blockIdx.x; // batch index

    int k = threadIdx.x + blockIdx.y * blockDim.x; // dim2 index
    if (k >= dim2) return;

    // Initialize max value and index
    float max_val = -FLT_MAX;
    int64_t max_idx = 0;

    for (int d = 0; d < dim1; d++) {
        int idx = n * dim1 * dim2 + d * dim2 + k;
        float val = input[idx];
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }

    // Write the index of the maximum value to output
    output[n * dim2 + k] = max_idx;
}

torch::Tensor argmax_dim1_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto dim1 = input.size(1);
    const auto dim2 = input.size(2);

    // Allocate output tensor
    auto output = torch::zeros({batch_size, dim2}, torch::dtype(torch::kInt64).device(input.device()));

    // Launch the CUDA kernel
    dim3 block(256);
    dim3 grid(batch_size, (dim2 + block.x - 1) / block.x);

    argmax_dim1_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        batch_size,
        dim1,
        dim2
    );

    return output;
}
'''

argmax_dim1_cpp_source = """
torch::Tensor argmax_dim1_cuda(torch::Tensor input);
"""

# Compile and load the extension module
argmax_module = load_inline(
    name='argmax_module',
    cpp_sources=argmax_dim1_cpp_source,
    cuda_sources=argmax_dim1_cuda_source,
    functions=['argmax_dim1_cuda'],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over dimension 1 using a custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax. Note: dim must be 1 for this implementation.

        Args:
            dim (int): The dimension to perform argmax over. Must be 1.
        """
        super(ModelNew, self).__init__()
        assert dim == 1, "This optimized model only supports argmax over dimension 1."
        self.dim = dim
        self.argmax_cuda = argmax_module.argmax_dim1_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over dimension 1 to the input tensor using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim1, dim2].

        Returns:
            torch.Tensor: Output tensor with argmax applied over dim=1, shape [batch_size, dim2].
        """
        return self.argmax_cuda(x.contiguous())