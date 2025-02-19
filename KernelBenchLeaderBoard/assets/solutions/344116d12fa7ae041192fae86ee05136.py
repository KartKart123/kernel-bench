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
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int out_h,
    const int out_w
) {
    const int thread_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    const int total_elements = batch_size * out_channels * out_h * out_w;
    
    for (int idx = thread_pos; idx < total_elements; idx += total_threads) {
        const int w_out = idx % out_w;
        const int h_out = (idx / out_w) % out_h;
        const int c_out = (idx / (out_w * out_h)) % out_channels;
        const int b = idx / (out_w * out_h * out_channels);
        
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    const int h_in = (h_out + pad_h - kh) / stride_h;
                    const int w_in = (w_out + pad_w - kw) / stride_w;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        if ((h_out + pad_h - kh) % stride_h == 0 && 
                            (w_out + pad_w - kw) % stride_w == 0) {
                            
                            const int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                            const int weight_idx = ((c_in * out_channels + c_out) * kernel_h + kh) * kernel_w + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    std::tuple<int, int> stride,
    std::tuple<int, int> padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int out_channels = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int stride_h = std::get<0>(stride);
    const int stride_w = std::get<1>(stride);
    const int pad_h = std::get<0>(padding);
    const int pad_w = std::get<1>(padding);
    
    const int out_h = (height - 1) * stride_h - 2 * pad_h + kernel_h;
    const int out_w = (width - 1) * stride_w - 2 * pad_w + kernel_w;
    
    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_h * out_w + threads - 1) / threads;
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        out_h,
        out_w
    );
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    std::tuple<int, int> stride,
    std::tuple<int, int> padding);
"""

conv_transpose2d_cuda = load_inline(
    name='conv_transpose2d_cuda',
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=['conv_transpose2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
        self.stride = stride
        self.padding = padding
        nn.init.kaiming_uniform_(self.weight, a=2.236)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose2d_cuda.conv_transpose2d_cuda(x, self.weight, self.stride, self.padding)