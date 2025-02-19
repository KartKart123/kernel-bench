import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for AvgPool3d
avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int input_D, int input_H, int input_W,
    int output_D, int output_H, int output_W,
    int kernel_size_D, int kernel_size_H, int kernel_size_W,
    int stride_D, int stride_H, int stride_W) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * output_D * output_H * output_W;

    if (index < total_elements) {
        int ow = index % output_W;
        int oh = (index / output_W) % output_H;
        int od = (index / (output_W * output_H)) % output_D;
        int c = (index / (output_W * output_H * output_D)) % C;
        int n = index / (output_W * output_H * output_D * C);

        int d_start = od * stride_D;
        int h_start = oh * stride_H;
        int w_start = ow * stride_W;

        float sum = 0.0f;
        for (int kd = 0; kd < kernel_size_D; ++kd) {
            for (int kh = 0; kh < kernel_size_H; ++kh) {
                for (int kw = 0; kw < kernel_size_W; ++kw) {
                    int id = d_start + kd;
                    int ih = h_start + kh;
                    int iw = w_start + kw;

                    if (id < input_D && ih < input_H && iw < input_W) {
                        int idx = (((n * C + c) * input_D + id) * input_H + ih) * input_W + iw;
                        sum += input[idx];
                    }
                }
            }
        }

        sum /= (kernel_size_D * kernel_size_H * kernel_size_W);

        int out_idx = index;
        output[out_idx] = sum;
    }
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size_D, int kernel_size_H, int kernel_size_W, int stride_D, int stride_H, int stride_W) {
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto input_D = input.size(2);
    const auto input_H = input.size(3);
    const auto input_W = input.size(4);

    const auto output_D = (input_D - kernel_size_D) / stride_D + 1;
    const auto output_H = (input_H - kernel_size_H) / stride_H + 1;
    const auto output_W = (input_W - kernel_size_W) / stride_W + 1;

    auto output = torch::empty({N, C, output_D, output_H, output_W}, input.options());

    int total_elements = N * C * output_D * output_H * output_W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, input_D, input_H, input_W,
        output_D, output_H, output_W,
        kernel_size_D, kernel_size_H, kernel_size_W,
        stride_D, stride_H, stride_W
    );

    return output;
}
"""

avg_pool3d_cpp_source = """
torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size_D, int kernel_size_H, int kernel_size_W, int stride_D, int stride_H, int stride_W);
"""

# Compile the inline CUDA code for AvgPool3d
avg_pool3d = load_inline(
    name='avg_pool3d',
    cpp_sources=avg_pool3d_cpp_source,
    cuda_sources=avg_pool3d_source,
    functions=['avg_pool3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool3d = avg_pool3d

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.avg_pool3d.avg_pool3d_cuda(x, 4, 4, 4, 4, 4, 4)
        return x