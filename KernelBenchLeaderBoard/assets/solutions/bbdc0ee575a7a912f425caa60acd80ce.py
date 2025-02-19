import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                     const float* __restrict__ mean, const float* __restrict__ var, 
                                     int batch_size, int num_features, int height, int width, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = batch_size * num_features * height * width;
    if (idx < num_elements) {
        int n = idx / (num_features * height * width);
        int c = (idx / (height * width)) % num_features;
        int hw = idx % (height * width);
        
        int mean_var_idx = n * num_features + c;
        float mean_val = mean[mean_var_idx];
        float var_val = var[mean_var_idx];
        
        output[idx] = (input[idx] - mean_val) / sqrtf(var_val + eps);
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor input, float eps) {
    auto batch_size = input.size(0);
    auto num_features = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto mean = input.mean({2, 3}, true);
    auto var = input.var({2, 3}, false, true);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_elements = batch_size * num_features * height * width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    instance_norm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), 
        mean.data_ptr<float>(), var.data_ptr<float>(), 
        batch_size, num_features, height, width, eps
    );

    return output;
}
"""

instance_norm_cpp_source = "torch::Tensor instance_norm_cuda(torch::Tensor input, float eps);"

# Compile the inline CUDA code for Instance Normalization
instance_norm = load_inline(
    name='instance_norm',
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=['instance_norm_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    """
    Optimized model that performs Instance Normalization using a custom CUDA kernel.
    """
    def __init__(self, num_features: int):
        """
        Initializes the custom InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies custom Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.instance_norm.instance_norm_cuda(x, self.eps)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]