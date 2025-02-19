import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom GELU CUDA code
gelu_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float y = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = y;
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaDeviceSynchronize();
    return output;
}
'''

gelu_cpp_source = '''
torch::Tensor gelu_cuda(torch::Tensor input);
'''

# Compile the custom GELU operator
gelu_op = load_inline(
    name='gelu_op',
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=['gelu_cuda'],
    verbose=False,
)

# Custom Global Average Pooling CUDA code
global_avg_pool_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void global_avg_pool_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int channels, int height, int width) {

    int b = blockIdx.x;
    int c = blockIdx.y;

    int hw = height * width;
    int idx = threadIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    while (idx < hw) {
        int offset = ((b * channels + c) * height * width) + idx;
        sum += input[offset];
        idx += blockDim.x;
    }

    // Reduce sum within block
    __shared__ float shared_sum[256];
    shared_sum[tid] = sum;
    __syncthreads();

    // Reduce sum in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b * channels + c] = shared_sum[0] / (float)(hw);
    }
}

torch::Tensor global_avg_pool_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    auto output = torch::zeros({batch_size, channels}, input.options());

    dim3 blocks(batch_size, channels);
    int threads = 256;

    global_avg_pool_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);
    cudaDeviceSynchronize();
    return output;
}
'''

global_avg_pool_cpp_source = '''
torch::Tensor global_avg_pool_cuda(torch::Tensor input);
'''

# Compile the custom Global Average Pooling operator
global_avg_pool_op = load_inline(
    name='global_avg_pool_op',
    cpp_sources=global_avg_pool_cpp_source,
    cuda_sources=global_avg_pool_source,
    functions=['global_avg_pool_cuda'],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for GELU activation and global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.gelu = gelu_op
        self.global_avg_pool = global_avg_pool_op

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu.gelu_cuda(x)
        x = self.global_avg_pool.global_avg_pool_cuda(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]