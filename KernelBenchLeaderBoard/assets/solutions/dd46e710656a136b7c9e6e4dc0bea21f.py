import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA code for Max Pooling 1D
maxpool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void maxpool1d_cuda_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      int batch_size, int channels, int input_length, int output_length,
                                      int kernel_size, int stride, int padding, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_length) return;

    int out_pos = idx % output_length;
    int c = (idx / output_length) % channels;
    int n = idx / (channels * output_length);

    int in_start = out_pos * stride - padding;
    float max_val = -FLT_MAX;
    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = in_start + k * dilation;
        if (in_pos >= 0 && in_pos < input_length) {
            int input_idx = n * channels * input_length + c * input_length + in_pos;
            float val = input[input_idx];
            if (val > max_val) max_val = val;
        }
    }
    int output_idx = n * channels * output_length + c * output_length + out_pos;
    output[output_idx] = max_val;
}

torch::Tensor maxpool1d_cuda_forward(torch::Tensor input, int kernel_size, int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);

    int output_length = (input_length + 2 * padding - dilation * (kernel_size -1) -1) / stride +1;

    auto output = torch::empty({batch_size, channels, output_length}, input.options());

    const int threads = 1024;
    const int total_threads = batch_size * channels * output_length;
    const int blocks = (total_threads + threads -1)/threads;

    maxpool1d_cuda_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation);
    return output;
}
"""

maxpool1d_cpp_source = """
torch::Tensor maxpool1d_cuda_forward(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);
"""

# Compile the custom CUDA kernel
maxpool1d = load_inline(
    name='maxpool1d',
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_cuda_source,
    functions=['maxpool1d_cuda_forward'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Model with custom CUDA MaxPool1d operator.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.maxpool1d_cuda = maxpool1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool1d_cuda.maxpool1d_cuda_forward(
            x, self.kernel_size, self.stride, self.padding, self.dilation)