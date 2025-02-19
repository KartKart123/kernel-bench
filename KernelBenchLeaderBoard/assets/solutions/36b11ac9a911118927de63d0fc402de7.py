import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matmul + min + subtraction
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(const float* input, const float* weight, const float* bias, float* output, const float constant, int in_features, int out_features, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * out_features) {
        int batch_index = i / out_features;
        int output_index = i % out_features;
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input[batch_index * in_features + k] * weight[output_index * in_features + k];
        }
        sum += bias[output_index];
        output[i] = fminf(sum, constant) - constant;
    }
}

torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant, int in_features, int out_features, int batch_size) {
    auto output = torch::zeros({batch_size, out_features}, input.options());
    const int block_size = 256;
    const int num_blocks = (batch_size * out_features + block_size - 1) / block_size;
    fused_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), constant, in_features, out_features, batch_size);
    return output;
}
"""

fused_kernel_cpp_source = "torch::Tensor fused_operation_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float constant, int in_features, int out_features, int batch_size);"

fused_kernel = load_inline(
    name='fused_kernel',
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=['fused_operation_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies minimum, and subtracts a constant using a fused CUDA kernel.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.fused_kernel = fused_kernel

    def forward(self, x):
        batch_size = x.shape[0]
        return self.fused_kernel.fused_operation_cuda(x, self.linear.weight, self.linear.bias, self.constant.item(), self.linear.in_features, self.linear.out_features, batch_size)