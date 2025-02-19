import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Triplet Margin Loss
triplet_margin_loss_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triplet_margin_loss_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ losses,
    int batch_size,
    int feature_size,
    float margin)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size)
        return;

    const float* anchor_sample = anchor + batch_idx * feature_size;
    const float* positive_sample = positive + batch_idx * feature_size;
    const float* negative_sample = negative + batch_idx * feature_size;

    // Compute squared L2 distances
    float dist_ap = 0.0f;
    float dist_an = 0.0f;
    for (int i = 0; i < feature_size; ++i)
    {
        float diff_ap = anchor_sample[i] - positive_sample[i];
        float diff_an = anchor_sample[i] - negative_sample[i];
        dist_ap += diff_ap * diff_ap;
        dist_an += diff_an * diff_an;
    }
    dist_ap = sqrtf(dist_ap);
    dist_an = sqrtf(dist_an);

    float loss = fmaxf(dist_ap - dist_an + margin, 0.0f);
    losses[batch_idx] = loss;
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin)
{
    auto batch_size = anchor.size(0);
    auto feature_size = anchor.size(1);
    auto losses = torch::empty({batch_size}, anchor.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    triplet_margin_loss_kernel<<<blocks, threads>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        losses.data_ptr<float>(),
        batch_size,
        feature_size,
        margin);

    return losses.mean();
}
"""

triplet_margin_loss_cpp_source = "torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);"

# Compile the inline CUDA code
triplet_margin_loss = load_inline(
    name='triplet_margin_loss',
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_source,
    functions=['triplet_margin_loss_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss with a custom CUDA kernel using the margin provided.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_margin_loss = triplet_margin_loss

    def forward(self, anchor, positive, negative):
        return self.triplet_margin_loss.triplet_margin_loss_cuda(anchor, positive, negative, self.margin)