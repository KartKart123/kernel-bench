import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Max Pooling 2D
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool2d_cuda_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_height * output_width;
    if (index >= total_elements)
        return;

    // Compute n, c, h_out, w_out
    int w_out = index % output_width;
    int h_out = (index / output_width) % output_height;
    int c = (index / (output_width * output_height)) % channels;
    int n = index / (channels * output_height * output_width);

    // Compute h_start and w_start for the pooling window
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    float max_val = -FLT_MAX;
    // Iterate over the pooling window
    for (int i = 0; i < kernel_size; ++i)
    {
        for (int j = 0; j < kernel_size; ++j)
        {
            int h_in = h_start + i * dilation;
            int w_in = w_start + j * dilation;

            // Check if h_in and w_in are within input bounds
            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width)
            {
                int input_index = n * channels * input_height * input_width
                                + c * input_height * input_width
                                + h_in * input_width
                                + w_in;
                float val = input[input_index];
                if (val > max_val)
                {
                    max_val = val;
                }
            }
        }
    }
    // Store the result
    int output_index = n * channels * output_height * output_width
                     + c * output_height * output_width
                     + h_out * output_width
                     + w_out;
    output[output_index] = max_val;
}

torch::Tensor max_pool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation) {
     // Get input dimensions
     int batch_size = x.size(0);
     int channels = x.size(1);
     int input_height = x.size(2);
     int input_width = x.size(3);

     // Compute output dimensions
     int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
     int output_width  = (input_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

     // Create output tensor
     auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
     auto output = torch::empty({batch_size, channels, output_height, output_width}, options);

     // Launch CUDA kernel
     // Compute total number of threads
     int total_threads = batch_size * channels * output_height * output_width;
     const int threads_per_block = 256;
     int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
     max_pool2d_cuda_kernel<<<blocks, threads_per_block>>> (
         x.data_ptr<float>(),
         output.data_ptr<float>(),
         batch_size,
         channels,
         input_height,
         input_width,
         output_height,
         output_width,
         kernel_size,
         stride,
         padding,
         dilation
     );
     return output;
}
"""

cpp_source = """
torch::Tensor max_pool2d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the inline CUDA code for Max Pooling 2D
max_pool2d_cuda = load_inline(
    name='max_pool2d_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['max_pool2d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.max_pool2d_cuda = max_pool2d_cuda
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        return self.max_pool2d_cuda.max_pool2d_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)