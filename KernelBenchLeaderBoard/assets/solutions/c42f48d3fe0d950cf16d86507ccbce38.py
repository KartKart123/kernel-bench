import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void softmax_forward_kernel(const float * __restrict__ input, float * __restrict__ output, int batch_size, int out_features)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x; // Batch index

    if (n < batch_size)
    {
        const float *input_row = input + n * out_features;
        float *output_row = output + n * out_features;

        // Find max value for numerical stability
        float max_val = input_row[0];
        for (int i = 1; i < out_features; ++i)
        {
            if (input_row[i] > max_val)
                max_val = input_row[i];
        }

        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < out_features; ++i)
        {
            float val = expf(input_row[i] - max_val);
            output_row[i] = val;
            sum += val;
        }

        // Normalize
        for (int i = 0; i < out_features; ++i)
        {
            output_row[i] /= sum;
        }
    }
}

torch::Tensor softmax_forward_cuda(torch::Tensor input)
{
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 128;
    const int blocks = (batch_size + threads - 1) / threads;

    softmax_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_features);

    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_forward_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for softmax
softmax_module = load_inline(
    name='custom_softmax',
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=['softmax_forward_cuda'],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized Model with custom CUDA softmax kernel
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = softmax_module.softmax_forward_cuda

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.bn(x)
        x = self.scale * x
        x = self.softmax(x)
        return x

batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    return [torch.randn(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]