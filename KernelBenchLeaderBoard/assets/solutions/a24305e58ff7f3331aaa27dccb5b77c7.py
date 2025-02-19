import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mean Squared Error
mse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = predictions[idx] - targets[idx];
        out[idx] = diff * diff;
    }
}

torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto out = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mse_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), size);

    return torch::mean(out);
}
"""

mse_cpp_source = "torch::Tensor mse_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code for Mean Squared Error
mse = load_inline(
    name='mse',
    cpp_sources=mse_cpp_source,
    cuda_sources=mse_source,
    functions=['mse_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse = mse

    def forward(self, predictions, targets):
        return self.mse.mse_cuda(predictions, targets)