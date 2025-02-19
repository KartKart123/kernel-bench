import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void online_softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    // Find max value for numerical stability
    float thread_max = -INFINITY;
    for(int i = tid; i < dim; i += blockDim.x) {
        thread_max = max(thread_max, input[batch_idx * dim + i]);
    }
    
    // Reduce max values within block
    __shared__ float shared_max_array[256];
    shared_max_array[tid] = thread_max;
    __syncthreads();
    
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            shared_max_array[tid] = max(shared_max_array[tid], shared_max_array[tid + stride]);
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        shared_max = shared_max_array[0];
    }
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for(int i = tid; i < dim; i += blockDim.x) {
        float val = exp(input[batch_idx * dim + i] - shared_max);
        output[batch_idx * dim + i] = val;
        thread_sum += val;
    }
    
    // Reduce sum within block
    __shared__ float shared_sum_array[256];
    shared_sum_array[tid] = thread_sum;
    __syncthreads();
    
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            shared_sum_array[tid] += shared_sum_array[tid + stride];
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        shared_sum = shared_sum_array[0];
    }
    __syncthreads();
    
    // Normalize
    for(int i = tid; i < dim; i += blockDim.x) {
        output[batch_idx * dim + i] /= shared_sum;
    }
}

torch::Tensor online_softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    
    const int threads = 256;
    const int blocks = batch_size;
    
    online_softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
    
    return output;
}
"""

softmax_cpp_source = """
torch::Tensor online_softmax_cuda(torch::Tensor input);
"""

online_softmax = load_inline(
    name='online_softmax',
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=['online_softmax_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.online_softmax = online_softmax
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_softmax.online_softmax_cuda(x.cuda())