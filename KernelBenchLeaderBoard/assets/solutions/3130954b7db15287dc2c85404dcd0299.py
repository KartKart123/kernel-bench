import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused avg_pool3d, scaling, and bias addition

fused_avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float total_scale,
    const float* __restrict__ adjusted_bias,
    const int N, const int C,
    const int input_D, const int input_H, const int input_W,
    const int output_D, const int output_H, const int output_W,
    const int kernel_D, const int kernel_H, const int kernel_W,
    const int stride_D, const int stride_H, const int stride_W,
    const int pad_D, const int pad_H, const int pad_W,
    const int input_sN, const int input_sC, const int input_sD, const int input_sH, const int input_sW,
    const int output_sN, const int output_sC, const int output_sD, const int output_sH, const int output_sW)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    int total_elements = N * C * output_D * output_H * output_W;

    for (int linear_idx = index; linear_idx < total_elements; linear_idx += total_threads) {
        int n = linear_idx / (C * output_D * output_H * output_W);
        int c = (linear_idx / (output_D * output_H * output_W)) % C;
        int od = (linear_idx / (output_H * output_W)) % output_D;
        int oh = (linear_idx / output_W) % output_H;
        int ow = linear_idx % output_W;

        float sum = 0.0f;
        int count = 0;

        int id_start = od * stride_D - pad_D;
        int ih_start = oh * stride_H - pad_H;
        int iw_start = ow * stride_W - pad_W;

        for (int kd = 0; kd < kernel_D; ++kd) {
            int id = id_start + kd;
            if (id >= 0 && id < input_D) {
                for (int kh = 0; kh < kernel_H; ++kh) {
                    int ih = ih_start + kh;
                    if (ih >= 0 && ih < input_H) {
                        for (int kw = 0; kw < kernel_W; ++kw) {
                            int iw = iw_start + kw;
                            if (iw >= 0 && iw < input_W) {
                                int input_idx = n * input_sN + c * input_sC + id * input_sD + ih * input_sH + iw * input_sW;
                                sum += input[input_idx];
                                ++count;
                            }
                        }
                    }
                }
            }
        }
        float avg = (count > 0) ? sum / count : 0.0f;
        avg = avg * total_scale + adjusted_bias[c];
        int output_idx = n * output_sN + c * output_sC + od * output_sD + oh * output_sH + ow * output_sW;
        output[output_idx] = avg;
    }
}

torch::Tensor fused_avg_pool3d_cuda(
    torch::Tensor input,
    float total_scale,
    torch::Tensor adjusted_bias,
    int kernel_D, int kernel_H, int kernel_W,
    int stride_D, int stride_H, int stride_W,
    int pad_D, int pad_H, int pad_W)
{
    // Get input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int input_D = input.size(2);
    int input_H = input.size(3);
    int input_W = input.size(4);

    // Calculate output dimensions
    int output_D = (input_D + 2 * pad_D - kernel_D) / stride_D + 1;
    int output_H = (input_H + 2 * pad_H - kernel_H) / stride_H + 1;
    int output_W = (input_W + 2 * pad_W - kernel_W) / stride_W + 1;

    auto output = torch::zeros({N, C, output_D, output_H, output_W}, input.options());

    int threads = 1024;
    int blocks = (N * C * output_D * output_H * output_W + threads - 1) / threads;

    // Get strides
    auto input_strides = input.strides();
    auto output_strides = output.strides();

    fused_avg_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_scale,
        adjusted_bias.data_ptr<float>(),
        N, C,
        input_D, input_H, input_W,
        output_D, output_H, output_W,
        kernel_D, kernel_H, kernel_W,
        stride_D, stride_H, stride_W,
        pad_D, pad_H, pad_W,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4]
    );

    return output;
}
"""

fused_avg_pool3d_cpp_source = """
torch::Tensor fused_avg_pool3d_cuda(
    torch::Tensor input,
    float total_scale,
    torch::Tensor adjusted_bias,
    int kernel_D, int kernel_H, int kernel_W,
    int stride_D, int stride_H, int stride_W,
    int pad_D, int pad_H, int pad_W);
"""

# Compile the inline CUDA code for fused avg_pool3d
fused_avg_pool3d = load_inline(
    name='fused_avg_pool3d',
    cpp_sources=fused_avg_pool3d_cpp_source,
    cuda_sources=fused_avg_pool3d_source,
    functions=['fused_avg_pool3d_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
)

class ModelNew(nn.Module):
    """
    Optimized Model with Custom CUDA Kernel for fused avg_pool3d, scaling, and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.kernel_size = 2  # AvgPool3d kernel size
        self.stride = 2       # AvgPool3d stride
        self.padding = 0      # AvgPool3d padding
        self.fused_avg_pool3d = fused_avg_pool3d

    def forward(self, x):
        x = self.conv_transpose(x)
        total_scale = self.scale1 * self.scale2
        adjusted_bias = self.bias * self.scale2

        x = self.fused_avg_pool3d.fused_avg_pool3d_cuda(
            x,
            total_scale.item(),
            adjusted_bias,
            self.kernel_size, self.kernel_size, self.kernel_size,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding
        )
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]