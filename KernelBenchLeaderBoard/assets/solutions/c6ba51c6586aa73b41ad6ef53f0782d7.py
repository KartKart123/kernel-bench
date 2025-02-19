import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for avg_pool2d
avg_pool2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void avg_pool2d_kernel(const float* input,
                                  float* output,
                                  int N, int C, int H_in, int W_in,
                                  int H_out, int W_out,
                                  int kernel_size, int stride, int padding)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int total = N * C * H_out * W_out;

    if (index < total) {
        int w_out_idx = index % W_out;
        int h_out_idx = (index / W_out) % H_out;
        int c_idx = (index / (W_out * H_out)) % C;
        int n_idx = index / (W_out * H_out * C);

        int h_start = h_out_idx * stride - padding;
        int w_start = w_out_idx * stride - padding;
        int h_end = h_start + kernel_size;
        int w_end = w_start + kernel_size;
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);
        h_end = min(h_end, H_in);
        w_end = min(w_end, W_in);

        float sum = 0.0;
        int count = 0;
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int input_idx = ((n_idx * C + c_idx) * H_in + h) * W_in + w;
                sum += input[input_idx];
                count += 1;
            }
        }
        output[index] = sum / count;
    }
}

torch::Tensor avg_pool2d_cuda_forward(torch::Tensor input,
                                      int kernel_size,
                                      int stride,
                                      int padding)
{
    auto N = input.size(0);
    auto C = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);

    int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({N, C, H_out, W_out}, input.options());

    int total = N * C * H_out * W_out;

    const int threads = 1024;
    const int blocks = (total + threads - 1) / threads;

    avg_pool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        kernel_size, stride, padding);

    return output;
}
"""

avg_pool2d_cpp_source = """
torch::Tensor avg_pool2d_cuda_forward(torch::Tensor input,
                                      int kernel_size,
                                      int stride,
                                      int padding);
"""

# Compile the inline CUDA code for avg_pool2d
avg_pool2d = load_inline(
    name='avg_pool2d',
    cpp_sources=avg_pool2d_cpp_source,
    cuda_sources=avg_pool2d_source,
    functions=['avg_pool2d_cuda_forward'],
    verbose=True,
    extra_cuda_cflags=['--expt-relaxed-constexpr'],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using custom CUDA kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool2d_cuda_forward = avg_pool2d.avg_pool2d_cuda_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg_pool2d_cuda_forward(x, self.kernel_size, self.stride, self.padding)