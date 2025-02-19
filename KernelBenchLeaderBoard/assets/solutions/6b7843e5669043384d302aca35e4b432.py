import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for KL Divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void kl_div_kernel(const float* log_predictions, const float* targets, float* out, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_classes) {
        int b = idx / num_classes;
        int c = idx % num_classes;
        float log_p = log_predictions[idx];
        float target = targets[idx];
        out[idx] = target * (log(target) - log_p);
    }
}

torch::Tensor kl_div_cuda(torch::Tensor log_predictions, torch::Tensor targets) {
    auto batch_size = log_predictions.size(0);
    auto num_classes = log_predictions.size(1);
    auto out = torch::zeros_like(log_predictions);

    const int block_size = 256;
    const int num_blocks = (batch_size * num_classes + block_size - 1) / block_size;

    kl_div_kernel<<<num_blocks, block_size>>>(log_predictions.data_ptr<float>(), targets.data_ptr<float>(), out.data_ptr<float>(), batch_size, num_classes);

    return out.sum(1).mean();
}
"""

kl_div_cpp_source = "torch::Tensor kl_div_cuda(torch::Tensor log_predictions, torch::Tensor targets);"

# Compile the inline CUDA code for KL Divergence
kl_div = load_inline(
    name='kl_div',
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=['kl_div_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div = kl_div

    def forward(self, predictions, targets):
        log_predictions = torch.log(predictions)
        return self.kl_div.kl_div_cuda(log_predictions, targets)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), torch.randn(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []