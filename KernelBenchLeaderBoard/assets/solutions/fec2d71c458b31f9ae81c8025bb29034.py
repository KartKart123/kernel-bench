import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mean_tanh_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void mean_tanh_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int n = idx / C;
        int c = idx % C;
        float sum = 0.0f;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int index = ((n * C + c) * H + h) * W + w;
                sum += input[index];
            }
        }
        float mean = sum / (H * W);
        float tanh_mean = tanhf(mean);
        output[idx] = tanh_mean;
    }
}

torch::Tensor mean_tanh_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::empty({N * C}, input.options());

    int total = N * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    mean_tanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);

    return output.view({N, C, 1, 1});
}
"""

mean_tanh_cpp_source = "torch::Tensor mean_tanh_cuda(torch::Tensor input);"

# Compile the inline CUDA code for mean and tanh
mean_tanh = load_inline(
    name='mean_tanh',
    cpp_sources=mean_tanh_cpp_source,
    cuda_sources=mean_tanh_source,
    functions=['mean_tanh_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.mean_tanh = mean_tanh

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh(x)
        x = self.mean_tanh.mean_tanh_cuda(x)
        return x