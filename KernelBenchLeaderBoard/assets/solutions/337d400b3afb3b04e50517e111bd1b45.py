import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA code for scaled residual addition
cuda_source = """
#include <torch/extension.h>

__global__ void scaled_residual_add_kernel(const float* y_linear, const float* y_linear_detached, float* output, int size, float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float y_lin = y_linear[idx];
        float y_lin_detached = y_linear_detached[idx];
        output[idx] = scaling_factor * y_lin + y_lin_detached;
    }
}

void scaled_residual_add(torch::Tensor y_linear, torch::Tensor y_linear_detached, torch::Tensor output, float scaling_factor) {
    int size = y_linear.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    scaled_residual_add_kernel<<<blocks, threads>>>(
        y_linear.data_ptr<float>(),
        y_linear_detached.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        scaling_factor
    );
}
"""

cpp_source = """
void scaled_residual_add(torch::Tensor y_linear, torch::Tensor y_linear_detached, torch::Tensor output, float scaling_factor);
"""

# Compile the custom CUDA code
scaled_residual_add_module = load_inline(
    name='scaled_residual_add_module',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['scaled_residual_add'],
    verbose=True
)

class ScaledResidualAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_linear, scaling_factor):
        y_linear_detached = y_linear.detach()
        output = torch.empty_like(y_linear)
        scaled_residual_add_module.scaled_residual_add(y_linear, y_linear_detached, output, scaling_factor)
        ctx.save_for_backward(y_linear, torch.tensor(scaling_factor))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y_linear, scaling_factor = ctx.saved_tensors
        grad_y_linear = grad_output * scaling_factor.item()
        return grad_y_linear, None

scaled_residual_add = ScaledResidualAddFunction.apply

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        x = scaled_residual_add(x, self.scaling_factor)
        return x

batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]