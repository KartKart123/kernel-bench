import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cosine similarity loss
cosine_similarity_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cosine_similarity_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ losses,
    const int batch_size,
    const int input_size
) {
    // Each block handles one sample in the batch
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    // Shared memory for reductions
    extern __shared__ float sdata[];

    // Pointers to data for this sample
    const float* pred = predictions + sample_idx * input_size;
    const float* targ = targets + sample_idx * input_size;

    // Intermediate sums for dot product and norms
    float thread_dot = 0.0f;
    float thread_pred_norm_sq = 0.0f;
    float thread_targ_norm_sq = 0.0f;

    for (int idx = threadIdx.x; idx < input_size; idx += blockDim.x) {
        float p = pred[idx];
        float t = targ[idx];
        thread_dot += p * t;
        thread_pred_norm_sq += p * p;
        thread_targ_norm_sq += t * t;
    }

    // Reduction for dot product
    sdata[threadIdx.x] = thread_dot;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float dot_product = sdata[0];

    // Reduction for pred_norm_sq
    sdata[threadIdx.x] = thread_pred_norm_sq;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float norm_pred = sqrtf(sdata[0] + 1e-8f);

    // Reduction for targ_norm_sq
    sdata[threadIdx.x] = thread_targ_norm_sq;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float norm_targ = sqrtf(sdata[0] + 1e-8f);

    if (threadIdx.x == 0) {
        float cosine_sim = dot_product / (norm_pred * norm_targ + 1e-8f);
        losses[sample_idx] = 1.0f - cosine_sim;
    }
}

torch::Tensor cosine_similarity_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const auto batch_size = predictions.size(0);
    const auto input_size = predictions.size(1);

    auto losses = torch::empty({batch_size}, predictions.options());

    const int threads = 256;
    const int blocks = batch_size;
    const size_t shared_mem_size = threads * sizeof(float);

    cosine_similarity_loss_kernel<<<blocks, threads, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        losses.data_ptr<float>(),
        batch_size,
        input_size
    );

    return losses;
}
"""

cosine_similarity_loss_cpp_source = """
torch::Tensor cosine_similarity_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code for cosine similarity loss
cosine_similarity_loss = load_inline(
    name='cosine_similarity_loss',
    cpp_sources=cosine_similarity_loss_cpp_source,
    cuda_sources=cosine_similarity_loss_source,
    functions=['cosine_similarity_loss_cuda'],
    verbose=True,
    extra_cflags=[],
    extra_cuda_cflags=[]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_similarity_loss = cosine_similarity_loss

    def forward(self, predictions, targets):
        losses = self.cosine_similarity_loss.cosine_similarity_loss_cuda(predictions, targets)
        return torch.mean(losses)