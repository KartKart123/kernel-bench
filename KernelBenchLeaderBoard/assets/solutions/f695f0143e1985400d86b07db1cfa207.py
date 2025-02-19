import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Smooth L1 (Huber) Loss
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* loss, int size, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = fabs(predictions[idx] - targets[idx]);
        if (diff < beta) {
            loss[idx] = 0.5 * diff * diff / beta;
        } else {
            loss[idx] = diff - 0.5 * beta;
        }
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta) {
    auto size = predictions.numel();
    auto loss = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), loss.data_ptr<float>(), size, beta);

    return loss.sum() / size;
}
"""

smooth_l1_loss_cpp_source = "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta);"

# Compile the inline CUDA code for Smooth L1 Loss
smooth_l1_loss = load_inline(
    name='smooth_l1_loss',
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=['smooth_l1_loss_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, beta=1.0):
        super(ModelNew, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss
        self.beta = beta

    def forward(self, predictions, targets):
        return self.smooth_l1_loss.smooth_l1_loss_cuda(predictions, targets, self.beta)