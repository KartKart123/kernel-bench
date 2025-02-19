import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cosine similarity
cosine_similarity_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_similarity_kernel(const float* predictions, const float* targets, float* cosine_sim, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float dot_product = 0.0f;
        float predictions_norm = 0.0f;
        float targets_norm = 0.0f;

        for (int i = 0; i < dim; i++) {
            dot_product += predictions[idx * dim + i] * targets[idx * dim + i];
            predictions_norm += predictions[idx * dim + i] * predictions[idx * dim + i];
            targets_norm += targets[idx * dim + i] * targets[idx * dim + i];
        }

        predictions_norm = sqrtf(predictions_norm);
        targets_norm = sqrtf(targets_norm);

        cosine_sim[idx] = dot_product / (predictions_norm * targets_norm);
    }
}

torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto dim = predictions.size(1);
    auto cosine_sim = torch::zeros({batch_size}, torch::kFloat32).cuda();

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    cosine_similarity_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), cosine_sim.data_ptr<float>(), batch_size, dim);

    return cosine_sim;
}
"""

cosine_similarity_cpp_source = "torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code for cosine similarity
cosine_similarity = load_inline(
    name='cosine_similarity',
    cpp_sources=cosine_similarity_cpp_source,
    cuda_sources=cosine_similarity_source,
    functions=['cosine_similarity_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_similarity = cosine_similarity

    def forward(self, predictions, targets):
        cosine_sim = self.cosine_similarity.cosine_similarity_cuda(predictions, targets)
        return torch.mean(1 - cosine_sim)