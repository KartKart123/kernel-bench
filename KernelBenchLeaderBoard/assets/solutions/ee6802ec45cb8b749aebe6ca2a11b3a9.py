import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cross-entropy loss
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void cross_entropy_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size)
        return;

    // Compute loss for sample i

    // find max_pred for numerical stability
    float max_pred = -INFINITY;
    for (int j = 0; j < num_classes; ++j) {
        float pred = predictions[i * num_classes + j];
        if (pred > max_pred) {
            max_pred = pred;
        }
    }

    // Compute log_sum_exp
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
        sum_exp += expf(predictions[i * num_classes + j] - max_pred);
    }
    float log_sum_exp = logf(sum_exp);

    int target_class = targets[i];

    float loss_i = - (predictions[i * num_classes + target_class] - max_pred) + log_sum_exp;
    losses[i] = loss_i;
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // predictions: [batch_size, num_classes]
    // targets: [batch_size]

    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);

    auto losses = torch::empty({batch_size}, predictions.options());

    // Launch kernel
    const int threads = 128;
    const int blocks = (batch_size + threads - 1) / threads;

    cross_entropy_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Return mean loss
    return losses.mean();
}
"""

cpp_source = """
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code for cross-entropy loss
cross_entropy = load_inline(
    name='cross_entropy',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['cross_entropy_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_include_paths=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy_cuda = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy_cuda.cross_entropy_cuda(predictions, targets)