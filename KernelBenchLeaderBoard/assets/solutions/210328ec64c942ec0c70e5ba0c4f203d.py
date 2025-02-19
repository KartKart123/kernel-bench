import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void frobenius_norm_kernel(const float* input, float* output, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / norm;
    }
}

__global__ void square_sum_kernel(const float* input, float* partial_sums, int size) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    while (gid < size) {
        float val = input[gid];
        sum += val * val;
        gid += blockDim.x * gridDim.x;
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

std::vector<torch::Tensor> frobenius_norm_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    auto partial_sums = torch::empty({num_blocks}, input.options());
    
    square_sum_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        size
    );
    
    float norm = std::sqrt(partial_sums.sum().item<float>());
    
    frobenius_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        norm,
        size
    );
    
    return {output};
}
"""

frobenius_norm_cpp_source = """
std::vector<torch::Tensor> frobenius_norm_cuda(torch::Tensor input);
"""

frobenius_norm = load_inline(
    name='frobenius_norm',
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=['frobenius_norm_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frobenius_norm.frobenius_norm_cuda(x)[0]