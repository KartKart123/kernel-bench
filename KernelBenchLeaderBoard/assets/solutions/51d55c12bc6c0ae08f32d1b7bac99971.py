import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cosine_sim_source = """
#include <torch/extension.h>
#include <cmath>

__global__ void cosine_similarity_kernel(const float* predictions, const float* targets, float* output, int batch_size, int input_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float dot_product = 0.0f;
        float norm_predictions = 0.0f;
        float norm_targets = 0.0f;
        for (int j = 0; j < input_size; ++j) {
            dot_product += predictions[i * input_size + j] * targets[i * input_size + j];
            norm_predictions += predictions[i * input_size + j] * predictions[i * input_size + j];
            norm_targets += targets[i * input_size + j] * targets[i * input_size + j];
        }
        norm_predictions = sqrtf(norm_predictions);
        norm_targets = sqrtf(norm_targets);
        if (norm_predictions > 0 && norm_targets > 0) {
            output[i] = dot_product / (norm_predictions * norm_targets);
        } else {
            output[i] = 0.0f; // Handle cases where norm is zero to avoid NaN
        }
    }
}


torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int input_size = predictions.size(1);
    auto output = torch::zeros({batch_size}, predictions.options());

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    cosine_similarity_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), output.data_ptr<float>(), batch_size, input_size);

    return output;
}
"""

cosine_sim_cpp_source = "torch::Tensor cosine_similarity_cuda(torch::Tensor predictions, torch::Tensor targets);"

cosine_sim_module = load_inline(
    name='cosine_sim_module',
    cpp_sources=cosine_sim_cpp_source,
    cuda_sources=cosine_sim_source,
    functions=['cosine_similarity_cuda'],
    verbose=True
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cosine_sim = cosine_sim_module

    def forward(self, predictions, targets):
        cosine_sim = self.cosine_sim.cosine_similarity_cuda(predictions, targets)
        return torch.mean(1 - cosine_sim)