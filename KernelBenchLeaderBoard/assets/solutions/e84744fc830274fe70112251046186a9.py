import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernels and functions for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int n = batch_size * out_channels * depth_out * height_out * width_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    int w_out = index % width_out;
    int h_index = index / width_out;
    int h_out = h_index % height_out;
    int d_index = h_index / height_out;
    int d_out = d_index % depth_out;
    int c_index = d_index / depth_out;
    int c_out = c_index % out_channels;
    int batch = c_index / out_channels;

    int g = c_out / (out_channels / groups); // Group index
    int in_c_start = g * (in_channels / groups);
    int in_c_end = in_c_start + (in_channels / groups);

    float value = 0.0f;

    for (int c_in = in_c_start; c_in < in_c_end; ++c_in) {
        for (int k_d = 0; k_d < kernel_d; ++k_d) {
            int z_in = d_out * stride_d - padding_d + k_d * dilation_d;
            if (z_in < 0 || z_in >= depth_in) continue;
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                int y_in = h_out * stride_h - padding_h + k_h * dilation_h;
                if (y_in < 0 || y_in >= height_in) continue;
                for (int k_w = 0; k_w < kernel_w; ++k_w) {
                    int x_in = w_out * stride_w - padding_w + k_w * dilation_w;
                    if (x_in < 0 || x_in >= width_in) continue;

                    int input_idx = ((batch * in_channels + c_in) * depth_in + z_in) * height_in * width_in + y_in * width_in + x_in;
                    int weight_idx = ((c_out * (in_channels / groups) + (c_in - in_c_start)) * kernel_d + k_d) * kernel_h * kernel_w + k_h * kernel_w + k_w;
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    if (bias != NULL) {
        value += bias[c_out];
    }

    int output_idx = ((batch * out_channels + c_out) * depth_out + d_out) * height_out * width_out + h_out * width_out + w_out;
    output[output_idx] = value;
}

torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    int depth_out = (depth_in + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int height_out = (height_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int width_out = (width_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = NULL;
    if (bias.defined() && bias.numel() > 0) {
        bias_ptr = bias.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    int n = batch_size * out_channels * depth_out * height_out * width_out;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    conv3d_forward_kernel<<<blocks, threads>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        depth_in,
        height_in,
        width_in,
        depth_out,
        height_out,
        width_out,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
        groups
    );

    return output;
}
"""

conv3d_forward_cpp_source = """
torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups
);
"""

conv3d_cuda = load_inline(
    name='conv3d_cuda',
    cpp_sources=conv3d_forward_cpp_source,
    cuda_sources=conv3d_source,
    functions=['conv3d_forward'],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Custom 3D convolution model using a custom CUDA kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel in the form (kernel_size_d, kernel_size_h, kernel_size_w).
        stride (tuple, optional): Stride of the convolution in the form (stride_d, stride_h, stride_w). Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input in the form (padding_d, padding_h, padding_w). Defaults to (0, 0, 0).
        dilation (tuple, optional): Spacing between kernel elements in the form (dilation_d, dilation_h, dilation_w). Defaults to (1, 1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d_cuda = conv3d_cuda
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = nn.Parameter(torch.Tensor())
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using the custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation

        return self.conv3d_cuda.conv3d_forward(
            x,
            self.weight,
            self.bias,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w,
            self.groups
        )