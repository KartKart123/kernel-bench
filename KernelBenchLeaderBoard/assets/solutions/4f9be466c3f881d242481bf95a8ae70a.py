import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_val = input[idx * dim];
        for (int i = 1; i < dim; i++) {
            if (input[idx * dim + i] > max_val) {
                max_val = input[idx * dim + i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += expf(input[idx * dim + i] - max_val);
        }
        for (int i = 0; i < dim; i++) {
            output[idx * dim + i] = expf(input[idx * dim + i] - max_val) / sum;
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

softmax_cpp_source = "torch::Tensor softmax_cuda(torch::Tensor input);"

# Compile the inline CUDA code for Softmax
softmax = load_inline(
    name='softmax',
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=['softmax_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax.softmax_cuda(x)