import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int out_depth,
    int out_height,
    int out_width)
{
    int o_channel = blockIdx.x;
    int batch = blockIdx.y;
    int output_idx_flat = blockIdx.z * blockDim.x + threadIdx.x;

    int od = output_idx_flat / (out_height * out_width);
    int oh = (output_idx_flat / out_width) % out_height;
    int ow = output_idx_flat % out_width;

    if (od >= out_depth || oh >= out_height || ow >= out_width) return;

    float output_value = 0.0;

    for (int i_channel = 0; i_channel < in_channels; ++i_channel) {
        for (int kd = 0; kd < kernel_depth; ++kd) {
            int id = od * stride_d - padding_d + kd;
            if (id < 0 || id >= input_depth) continue;
            for (int kh = 0; kh < kernel_height; ++kh) {
                int ih = oh * stride_h - padding_h + kh;
                if (ih < 0 || ih >= input_height) continue;
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int iw = ow * stride_w - padding_w + kw;
                    if (iw < 0 || iw >= input_width) continue;
                    // Compute indices
                    int input_idx = ((batch * in_channels + i_channel) * input_depth + id) * input_height * input_width + ih * input_width + iw;
                    int weight_idx = ((o_channel * in_channels + i_channel) * kernel_depth + kd) * kernel_height * kernel_width + kh * kernel_width + kw;
                    output_value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((batch * out_channels + o_channel) * out_depth + od) * out_height * out_width + oh * out_width + ow;
    output[output_idx] = output_value;
}

torch::Tensor conv3d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w)
{
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);

    int out_depth = (input_depth + 2 * padding_d - kernel_depth) / stride_d + 1;
    int out_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    const int total_threads = out_depth * out_height * out_width;
    const dim3 blocks(out_channels, batch_size, (total_threads + 1023) / 1024);
    const dim3 threads(1024);

    conv3d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        out_depth,
        out_height,
        out_width
    );

    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w);
"""

# Compile the custom CUDA code
conv3d_cuda = load_inline(
    name='conv3d_cuda',
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=['conv3d_cuda_forward'],
    verbose=True,
)

class Conv3dCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding):
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        output = conv3d_cuda.conv3d_cuda_forward(
            input.contiguous(), weight.contiguous(),
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2]
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        # Gradients w.r.t input
        grad_input = torch.nn.grad.conv3d_input(
            input_size=input.shape,
            weight=weight,
            grad_output=grad_output,
            stride=stride,
            padding=padding,
        )

        # Gradients w.r.t weight
        grad_weight = torch.nn.grad.conv3d_weight(
            input=input,
            weight_size=weight.shape,
            grad_output=grad_output,
            stride=stride,
            padding=padding,
        )

        return grad_input, grad_weight, None, None

class Conv3dCustom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(Conv3dCustom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels,
            *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = Conv3dCustomFunction.apply(input, self.weight, self.stride, self.padding)
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
        return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Note: Dilation and groups are not supported in this custom convolution
        if dilation != 1 or groups != 1:
            raise NotImplementedError("Dilation and groups are not implemented in custom Conv3d")
        self.conv3d = Conv3dCustom(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv3d(x)