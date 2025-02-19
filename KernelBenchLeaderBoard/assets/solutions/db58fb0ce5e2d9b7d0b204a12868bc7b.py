import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

smooth_l1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_kernel(const float* pred, const float* target, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        float abs_diff = abs(diff);
        if (abs_diff < 1.0f) {
            out[idx] = 0.5f * diff * diff;
        } else {
            out[idx] = abs_diff - 0.5f;
        }
    }
}

torch::Tensor smooth_l1_cuda(torch::Tensor pred, torch::Tensor target) {
    auto size = pred.numel();
    auto out = torch::zeros_like(pred);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    smooth_l1_kernel<<<num_blocks, block_size>>>(
        pred.data_ptr<float>(),
        target.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    return out.mean();
}
"""

smooth_l1_cpp_source = "torch::Tensor smooth_l1_cuda(torch::Tensor pred, torch::Tensor target);"

smooth_l1_cuda = load_inline(
    name='smooth_l1',
    cpp_sources=smooth_l1_cpp_source,
    cuda_sources=smooth_l1_source,
    functions=['smooth_l1_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.smooth_l1 = smooth_l1_cuda

    def forward(self, predictions, targets):
        return self.smooth_l1.smooth_l1_cuda(predictions, targets)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []