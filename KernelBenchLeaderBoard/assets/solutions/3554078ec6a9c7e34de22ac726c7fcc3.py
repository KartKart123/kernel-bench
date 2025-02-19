import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rms_norm_kernel(const float* x, float* out, int batch_size, int num_features, int size, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * size) {
        int batch_idx = idx / size;
        int feature_idx = idx % size;
        int feature_dim_idx = feature_idx / (size / num_features);
        int spatial_idx = feature_idx % (size / num_features);

        float sum_squares = 0.0f;
        for (int i = 0; i < num_features; i++) {
            sum_squares += x[batch_idx * num_features * (size / num_features) + i * (size / num_features) + spatial_idx] * x[batch_idx * num_features * (size / num_features) + i * (size / num_features) + spatial_idx];
        }
        float rms = sqrtf(sum_squares / num_features + eps);

        out[idx] = x[idx] / rms;
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, int num_features, float eps) {
    auto batch_size = x.size(0);
    auto size = x.numel() / batch_size;
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (batch_size * size + block_size - 1) / block_size;

    rms_norm_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, num_features, size, eps);

    return out;
}
"""

rms_norm_cpp_source = (
    "torch::Tensor rms_norm_cuda(torch::Tensor x, int num_features, float eps);"
)

rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.rms_norm = rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm.rms_norm_cuda(x, self.num_features, self.eps)