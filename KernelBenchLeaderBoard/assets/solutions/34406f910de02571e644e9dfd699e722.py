import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool2d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels, 
    const int height,
    const int width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * channels * out_height * out_width) return;
    
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    float maxval = -1e38;
    
    for(int kh = 0; kh < kernel_size; kh++) {
        for(int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_out * stride - padding + kh * dilation;
            int w_in = w_out * stride - padding + kw * dilation;
            
            if(h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                int in_idx = ((b * channels + c) * height + h_in) * width + w_in;
                maxval = max(maxval, input[in_idx]);
            }
        }
    }
    
    output[idx] = maxval;
}

torch::Tensor maxpool2d_cuda(
    torch::Tensor input,
    const int kernel_size,
    const int stride, 
    const int padding,
    const int dilation
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    
    auto out_height = ((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    auto out_width = ((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    
    auto output = torch::zeros({batch_size, channels, out_height, out_width}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * channels * out_height * out_width + threads - 1) / threads;
    
    maxpool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width, 
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_cuda(
    torch::Tensor input,
    const int kernel_size,
    const int stride,
    const int padding, 
    const int dilation);
"""

maxpool2d = load_inline(
    name='maxpool2d',
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_cuda_source,
    functions=['maxpool2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.maxpool2d = maxpool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d.maxpool2d_cuda(
            x.cuda(), 
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )