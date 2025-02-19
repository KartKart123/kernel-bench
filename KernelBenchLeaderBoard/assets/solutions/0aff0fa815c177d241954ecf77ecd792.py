import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused bias addition and activations
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void fused_bias_activation(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int batch_size,
    int features) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = batch_size * features;
    if (idx < num_elements) {
        int feature_idx = idx % features;
        float val = x[idx] + bias[feature_idx];
        // Apply Hardtanh activation
        val = fminf(fmaxf(val, -1.0f), 1.0f);
        // Apply Mish activation: mish(x) = x * tanh(softplus(x))
        float sp = logf(1.0f + expf(val)); // softplus
        float mish_val = val * tanhf(sp);
        y[idx] = mish_val;
    }
}

torch::Tensor fused_bias_activation_cuda(torch::Tensor x, torch::Tensor bias) {
    int batch_size = x.size(0);
    int features = x.size(1);
    auto y = torch::empty_like(x);

    int num_elements = batch_size * features;

    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    fused_bias_activation<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        features
    );

    return y;
}
"""

cpp_source = """
torch::Tensor fused_bias_activation_cuda(torch::Tensor x, torch::Tensor bias);
"""

# Compile the CUDA extension
fused_bias_activation = load_inline(
    name='fused_bias_activation',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_bias_activation_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA fused operator.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.fused_bias_activation = fused_bias_activation

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = self.fused_bias_activation.fused_bias_activation_cuda(x, self.bias)
        x = self.groupnorm(x)
        return x

batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]