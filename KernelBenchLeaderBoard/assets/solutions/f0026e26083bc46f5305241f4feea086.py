import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cosine similarity
cosine_similarity_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_similarity_kernel(const float* predictions, const float* targets, float* out, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float dot_product = 0.0f;
        float norm_pred = 0.0f;
        float norm_target = 0.0f;
        for (int i = 0; i < dim; i++) {
            dot_product += predictions[idx * dim + i] * targets[idx * dim + i];
            norm_pred += predictions[idx * dim + i] * predictions[idx * dim + i];
            norm_target += targets[idx * dim + i] * targets[idx * dim + i];
        }
        norm_pred = sqrt(norm_pred);
        norm_target = sqrt(norm_target);
        out[idx] = dot_product / (norm_pred * norm_target);
    }
}

torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto dim = predictions.size(1);
    auto out = torch::zeros_like(predictions.select(1, 0));

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    cosine_similarity_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    return out;
}

torch::Tensor cosine_similarity_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto cosine_sim = cosine_similarity_cuda(predictions, targets);
    return torch::mean(1 - cosine_sim);
}
"""

cosine_similarity_cpp_source = (
    "torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets);\n"
    "torch::Tensor cosine_similarity_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for cosine similarity
cosine_similarity = load_inline(
    name="cosine_similarity",
    cpp_sources=cosine_similarity_cpp_source,
    cuda_sources=cosine_similarity_source,
    functions=["cosine_similarity_cuda", "cosine_similarity_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cosine_similarity = cosine_similarity

    def forward(self, predictions, targets):
        return self.cosine_similarity.cosine_similarity_loss_cuda(predictions, targets)