import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels, 
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int out_padding_h,
    const int out_padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups
) {
    const int h_out = blockIdx.y;
    const int w_out = blockIdx.x;
    const int batch_idx = blockIdx.z / out_channels;
    const int out_ch = blockIdx.z % out_channels;
    
    if (h_out >= height_out || w_out >= width_out || batch_idx >= batch_size)
        return;

    float sum = 0.0f;
    const int in_ch_per_group = in_channels / groups;
    const int out_ch_per_group = out_channels / groups;
    const int group = out_ch / out_ch_per_group;
    
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            const int h_in = (h_out + padding_h - kh * dilation_h) / stride_h;
            const int w_in = (w_out + padding_w - kw * dilation_w) / stride_w;
            
            if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                for (int in_ch = group * in_ch_per_group; 
                     in_ch < (group + 1) * in_ch_per_group; 
                     in_ch++) {
                    
                    const float input_val = input[
                        batch_idx * in_channels * height_in * width_in +
                        in_ch * height_in * width_in +
                        h_in * width_in +
                        w_in
                    ];
                    
                    const float weight_val = weight[
                        in_ch * out_ch_per_group * kernel_h * kernel_w +
                        (out_ch % out_ch_per_group) * kernel_h * kernel_w +
                        kh * kernel_w +
                        kw
                    ];
                    
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    output[
        batch_idx * out_channels * height_out * width_out +
        out_ch * height_out * width_out +
        h_out * width_out +
        w_out
    ] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    std::tuple<int, int> stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> output_padding,
    std::tuple<int, int> dilation,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const auto [stride_h, stride_w] = stride;
    const auto [padding_h, padding_w] = padding;
    const auto [out_padding_h, out_padding_w] = output_padding;
    const auto [dilation_h, dilation_w] = dilation;
    
    const int height_out = (height_in - 1) * stride_h - 2 * padding_h + 
                          dilation_h * (kernel_h - 1) + out_padding_h + 1;
    const int width_out = (width_in - 1) * stride_w - 2 * padding_w + 
                         dilation_w * (kernel_w - 1) + out_padding_w + 1;
    
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, 
                             input.options());
    
    const dim3 blocks(width_out, height_out, batch_size * out_channels);
    const dim3 threads(1);
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in, 
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        out_padding_h,
        out_padding_w,
        dilation_h,
        dilation_w,
        groups
    );
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    std::tuple<int, int> stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> output_padding,
    std::tuple<int, int> dilation,
    int groups
);
"""

conv_transpose2d_cuda = load_inline(
    name='conv_transpose2d_cuda',
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=['conv_transpose2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1), 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, 
                                              kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        nn.init.kaiming_uniform_(self.weight, a=2.236)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_cuda.conv_transpose2d_cuda(
            x, self.weight, self.stride, self.padding,
            self.output_padding, self.dilation, self.groups
        )