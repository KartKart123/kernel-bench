import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

kl_div_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(const float* predictions, const float* targets, float* output, 
                             const int batch_size, const int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0f;
        for(int i = 0; i < dim; i++) {
            int index = idx * dim + i;
            if (targets[index] > 0) {
                sum += targets[index] * (log(targets[index]) - log(predictions[index]));
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto dim = predictions.size(1);
    
    auto output = torch::zeros({batch_size}, predictions.options());
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    kl_div_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
    
    return output.mean();
}
"""

kl_div_cpp_source = """
torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

kl_div_cuda = load_inline(
    name='kl_div_cuda',
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_cuda_source,
    functions=['kl_div_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.kl_div = kl_div_cuda

    def forward(self, predictions, targets):
        return self.kl_div.kl_div_cuda(predictions, targets)