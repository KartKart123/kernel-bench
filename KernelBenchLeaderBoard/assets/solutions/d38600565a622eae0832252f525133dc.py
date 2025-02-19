import torch
import torch.nn as nn
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
        float norm_pred = 0.0f;
        float norm_target = 0.0f;

        for (int i = 0; i < dim; ++i) {
            float pred_val = predictions[idx * dim + i];
            float target_val = targets[idx * dim + i];
            dot_product += pred_val * target_val;
            norm_pred += pred_val * pred_val;
            norm_target += target_val * target_val;
        }

        norm_pred = sqrtf(norm_pred);
        norm_target = sqrtf(norm_target);

        if (norm_pred == 0.0f || norm_target == 0.0f) {
            cosine_sim[idx] = 0.0f;
        } else {
            cosine_sim[idx] = dot_product / (norm_pred * norm_target);
        }
    }
}

torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto dim = predictions.size(1);
    auto cosine_sim = torch::zeros({batch_size}, predictions.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    cosine_similarity_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        cosine_sim.data_ptr<float>(),
        batch_size,
        dim
    );

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
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_similarity = cosine_similarity

    def forward(self, predictions, targets):
        cosine_sim = self.cosine_similarity.cosine_similarity_cuda(predictions, targets)
        return torch.mean(1 - cosine_sim)