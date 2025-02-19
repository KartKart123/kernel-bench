import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code
frobenius_norm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sum_of_squares_kernel(const float *x, float *sum, int size) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;

    while (i < size) {
        float val = x[i];
        temp += val * val;
        i += blockDim.x * gridDim.x;
    }

    cache[tid] = temp;

    __syncthreads();

    // Reduction in shared memory
    int offset = blockDim.x / 2;
    while (offset > 0) {
        if (tid < offset) {
            cache[tid] += cache[tid + offset];
        }
        __syncthreads();
        offset /= 2;
    }

    if (tid == 0) {
        atomicAdd(sum, cache[0]);
    }
}

__global__ void divide_by_norm_kernel(const float *x, float *out, float norm, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    while (i < size) {
        out[i] = x[i] / norm;
        i += blockDim.x * gridDim.x;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor x) {
    // Flatten x
    auto x_flat = x.contiguous().view(-1);
    auto size = x_flat.numel();

    // Allocate sum tensor on device
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto sum = torch::zeros(1, options);

    // Define block and grid sizes
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // Launch sum_of_squares_kernel
    sum_of_squares_kernel<<<blocks, threads>>>(x_flat.data_ptr<float>(), sum.data_ptr<float>(), size);

    // Synchronize to ensure kernel has finished
    cudaDeviceSynchronize();

    // Get sum from device
    float sum_host = sum.cpu().item<float>();

    // Compute norm
    float norm = sqrtf(sum_host);

    // Launch divide_by_norm_kernel
    auto out = torch::empty_like(x_flat);
    divide_by_norm_kernel<<<blocks, threads>>>(x_flat.data_ptr<float>(), out.data_ptr<float>(), norm, size);

    // Reshape out to original shape
    auto output = out.view_as(x);

    return output;
}
"""

# C++ source code
frobenius_norm_cpp_source = """
torch::Tensor frobenius_norm_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
frobenius_norm = load_inline(
    name='frobenius_norm',
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_cuda_source,
    functions=['frobenius_norm_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x):
        return self.frobenius_norm.frobenius_norm_cuda(x)