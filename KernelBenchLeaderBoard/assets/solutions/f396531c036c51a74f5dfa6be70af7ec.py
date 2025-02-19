import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel
cpp_source = """
torch::Tensor fused_kernel_cuda(torch::Tensor x1, torch::Tensor subtract, torch::Tensor original_x);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__device__ float gelu_cuda(float x) {
    // Approximation of the GELU activation function
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void fused_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ subtract,
    const float* __restrict__ original_x,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    int i = blockIdx.x;  // Batch index

    if (i >= batch_size)
        return;

    // Shared memory for reduction
    __shared__ float sdata[THREADS_PER_BLOCK];
    float sum = 0.0f;

    // Compute partial sum
    for (int j = threadIdx.x; j < out_features; j += blockDim.x) {
        int idx = i * out_features + j;
        float val = x1[idx] - subtract[j];
        sum += val;
    }

    // Store partial sum in shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduction to compute total sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    // Thread 0 computes mean and applies GELU
    float x_i = 0.0f;
    if (threadIdx.x == 0) {
        float mean = sdata[0] / out_features;
        x_i = gelu_cuda(mean);
        sdata[0] = x_i;  // Store x_i in shared memory
    }

    __syncthreads();
    x_i = sdata[0];

    // Compute output = x_i + original_x
    for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
        int out_idx = i * in_features + k;
        output[out_idx] = x_i + original_x[out_idx];
    }
}

torch::Tensor fused_kernel_cuda(torch::Tensor x1, torch::Tensor subtract, torch::Tensor original_x)
{
    int batch_size = x1.size(0);
    int out_features = x1.size(1);
    int in_features = original_x.size(1);

    auto output = torch::empty_like(original_x);

    const int threads = THREADS_PER_BLOCK;
    const int blocks = batch_size;

    fused_kernel<<<blocks, threads>>>(
        x1.data_ptr<float>(),
        subtract.data_ptr<float>(),
        original_x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

# Compile the CUDA code
fused_kernel_module = load_inline(
    name='fused_kernel',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_kernel_cuda'],
    extra_cuda_cflags=['-O3'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias).cuda()
        self.subtract = nn.Parameter(torch.randn(out_features).cuda())
        self.fused_kernel = fused_kernel_module

    def forward(self, x):
        x = x.cuda()
        original_x = x
        x1 = self.gemm(x)

        # Ensure tensors are contiguous
        x1 = x1.contiguous()
        subtract = self.subtract.contiguous()
        original_x = original_x.contiguous()

        # Call the fused CUDA kernel
        output = self.fused_kernel.fused_kernel_cuda(x1, subtract, original_x)

        return output

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]