import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* predictions, const long* targets, 
                                   float* losses, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_val = -INFINITY;
        for(int c = 0; c < num_classes; c++) {
            max_val = max(max_val, predictions[idx * num_classes + c]);
        }
        
        float sum = 0.0f;
        for(int c = 0; c < num_classes; c++) {
            sum += expf(predictions[idx * num_classes + c] - max_val);
        }
        float log_sum = logf(sum);
        
        int target = targets[idx];
        losses[idx] = -(predictions[idx * num_classes + target] - max_val - log_sum);
    }
}

torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto num_classes = predictions.size(1);
    auto losses = torch::empty({batch_size}, predictions.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    cross_entropy_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<long>(),
        losses.data_ptr<float>(),
        batch_size,
        num_classes
    );
    
    return losses.mean();
}
"""

cross_entropy_cpp_source = """
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

cross_entropy_cuda = load_inline(
    name='cross_entropy',
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_source,
    functions=['cross_entropy_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy_cuda

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_cuda(predictions, targets)