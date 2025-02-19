import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels, 
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_height,
    const int output_width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * channels * output_height * output_width) return;
    
    const int w_out = idx % output_width;
    const int h_out = (idx / output_width) % output_height;
    const int c = (idx / (output_width * output_height)) % channels;
    const int b = idx / (output_width * output_height * channels);

    float sum = 0.0f;
    int count = 0;
    
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;
    
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            const int h_in = h_start + kh;
            const int w_in = w_start + kw;
            
            if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                sum += input[((b * channels + c) * height + h_in) * width + w_in];
                count++;
            }
        }
    }
    
    output[idx] = sum / count;
}

torch::Tensor avg_pool2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    if (stride <= 0) stride = kernel_size;
    
    const int output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());
    
    const int total_threads = batch_size * channels * output_height * output_width;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;
    
    avg_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width, 
        kernel_size,
        stride,
        padding,
        output_height,
        output_width
    );
    
    return output;
}
"""

avg_pool2d_cpp_source = """
torch::Tensor avg_pool2d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride, 
    int padding
);
"""

avg_pool2d = load_inline(
    name='avg_pool2d',
    cpp_sources=avg_pool2d_cpp_source,
    cuda_sources=avg_pool2d_source,
    functions=['avg_pool2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.avg_pool2d = avg_pool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool2d.avg_pool2d_cuda(
            x.cuda(), 
            self.kernel_size,
            self.stride,
            self.padding
        )