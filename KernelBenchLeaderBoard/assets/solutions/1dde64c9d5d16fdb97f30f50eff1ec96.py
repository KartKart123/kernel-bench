import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int hw = height * width;
    const int chw = channels * hw;
    
    for (int index = tid; index < batch_size * channels * height * width; index += stride) {
        const int n = index / chw;
        const int c = (index % chw) / hw;
        const int pos = index % hw;
        
        // Compute mean and variance for each channel in each batch
        float sum = 0.0f;
        float sq_sum = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < height * width; i++) {
            float val = input[n * chw + c * hw + i];
            sum += val;
            sq_sum += val * val;
        }
        
        const float mean = sum / hw;
        const float variance = (sq_sum / hw) - (mean * mean);
        const float std = sqrt(variance + 1e-5f);
        
        // Normalize
        const float val = input[n * chw + c * hw + pos];
        output[index] = (val - mean) / std;
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (batch_size * channels * height * width + threads - 1) / threads;
    
    instance_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels, 
        height,
        width
    );
    
    return output;
}
"""

instance_norm_cpp_source = """
torch::Tensor instance_norm_cuda(torch::Tensor input);
"""

instance_norm = load_inline(
    name='instance_norm',
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=['instance_norm_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_cuda(x.cuda())