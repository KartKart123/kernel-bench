import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSoftmax
log_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void log_softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int b = idx / dim;
        int d = idx % dim;

        float max_val = -INFINITY;
        for (int i = 0; i < dim; ++i) {
            float val = input[b * dim + i];
            if (val > max_val) {
                max_val = val;
            }
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum_exp += expf(input[b * dim + i] - max_val);
        }

        output[idx] = input[idx] - max_val - logf(sum_exp);
    }
}

torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    auto batch_size = input.size(0);
    auto dim_size = input.size(1);
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (batch_size * dim_size + block_size - 1) / block_size;

    log_softmax_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim_size);

    return output;
}
"""

log_softmax_cpp_source = "torch::Tensor log_softmax_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code for LogSoftmax
log_softmax = load_inline(
    name='log_softmax',
    cpp_sources=log_softmax_cpp_source,
    cuda_sources=log_softmax_source,
    functions=['log_softmax_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.log_softmax = log_softmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_softmax.log_softmax_cuda(x, self.dim)