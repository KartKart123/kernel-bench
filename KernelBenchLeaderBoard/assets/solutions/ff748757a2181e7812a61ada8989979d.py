import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cosine similarity
cosine_similarity_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cosine_similarity_kernel(const float* predictions, const float* targets, float* output, int size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float dot_product = 0.0;
        float norm_pred = 0.0;
        float norm_target = 0.0;
        for (int i = 0; i < dim; ++i) {
            float pred_val = predictions[idx * dim + i];
            float target_val = targets[idx * dim + i];
            dot_product += pred_val * target_val;
            norm_pred += pred_val * pred_val;
            norm_target += target_val * target_val;
        }
        output[idx] = dot_product / (sqrtf(norm_pred) * sqrtf(norm_target) + 1e-8);
    }
}

torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets, int dim) {
    auto batch_size = predictions.size(0);
    auto out = torch::empty({batch_size}, predictions.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    cosine_similarity_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        dim
    );

    return out;
}
"""

cosine_similarity_cpp_source = "torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets, int dim);"

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
    A model that computes Cosine Similarity Loss for comparing vectors using a custom CUDA kernel.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_similarity = cosine_similarity

    def forward(self, predictions, targets):
        cosine_sim = self.cosine_similarity.cosine_similarity_cuda(predictions, targets, predictions.size(1))
        return torch.mean(1 - cosine_sim)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).cuda(), torch.randn(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return []