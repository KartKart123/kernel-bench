import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + Mish + Mish
linear_mish_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float softplusf(float x)
{
    return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}

__device__ float mishf(float x)
{
    float sp = softplusf(x);
    return x * tanhf(sp);
}

__global__ void linear_mish_kernel(const float* __restrict__ x,
                                   const float* __restrict__ W,
                                   const float* __restrict__ b,
                                   float* __restrict__ y,
                                   int batch_size, int in_features, int out_features)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Output feature index

    if (row < batch_size && col < out_features)
    {
        float temp = b[col];

        // Compute dot product
        for (int i = 0; i < in_features; ++i)
        {
            temp += x[row * in_features + i] * W[col * in_features + i];
        }

        // Apply Mish activation twice
        temp = mishf(mishf(temp));

        y[row * out_features + col] = temp;
    }
}

torch::Tensor linear_mish_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b)
{
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = W.size(0);

    auto y = torch::zeros({batch_size, out_features}, x.options());

    const int threads = 16;
    dim3 block_size(threads, threads);
    dim3 grid_size((out_features + threads - 1) / threads, (batch_size + threads - 1) / threads);

    linear_mish_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size, in_features, out_features
    );

    return y;
}
'''

linear_mish_cpp_source = '''
torch::Tensor linear_mish_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b);
'''

# Compile the inline CUDA code for fused Linear + Mish + Mish
linear_mish = load_inline(
    name='linear_mish',
    cpp_sources=[linear_mish_cpp_source],
    cuda_sources=[linear_mish_source],
    functions=['linear_mish_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear_mish = linear_mish  # Custom CUDA module

    def forward(self, x):
        # Ensure inputs are contiguous and on CUDA
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous().cuda()
        b = self.linear.bias.contiguous().cuda()

        # Use the custom fused Linear + Mish + Mish CUDA function
        x = self.linear_mish.linear_mish_cuda(x, W, b)
        return x

batch_size = 128
in_features = 10
out_features = 20

def get_inputs():
    return [torch.randn(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]