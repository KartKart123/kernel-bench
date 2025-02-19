import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused AvgPool3d and GELU activation
avgpool3d_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avgpool3d_gelu_kernel(const float* __restrict__ input, float* output,
                                      int batch_size, int channels, int in_d, int in_h, int in_w,
                                      int out_d, int out_h, int out_w,
                                      int kernel_d, int kernel_h, int kernel_w,
                                      int stride_d, int stride_h, int stride_w,
                                      int padding_d, int padding_h, int padding_w) {
    // Calculate output index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;

    if (index >= total_elements) return;

    // Compute n, c, od, oh, ow indices
    int ow = index % out_w;
    int tmp = index / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int od = tmp % out_d;
    tmp = tmp / out_d;
    int c = tmp % channels;
    int n = tmp / channels;

    // Compute input region
    int id_start = od * stride_d - padding_d;
    int ih_start = oh * stride_h - padding_h;
    int iw_start = ow * stride_w - padding_w;

    float sum = 0.0f;
    int count = 0;

    for (int kd = 0; kd < kernel_d; ++kd) {
        int id = id_start + kd;
        if (id < 0 || id >= in_d) continue;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = ih_start + kh;
            if (ih < 0 || ih >= in_h) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = iw_start + kw;
                if (iw < 0 || iw >= in_w) continue;

                int idx = (((n * channels + c) * in_d + id) * in_h + ih) * in_w + iw;
                sum += input[idx];
                ++count;
            }
        }
    }

    float val = sum / count;

    // Apply GELU activation
    val = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));

    // Write output
    int out_idx = index;
    output[out_idx] = val;
}

torch::Tensor avgpool3d_gelu_cuda(torch::Tensor input,
                                  int kernel_d, int kernel_h, int kernel_w,
                                  int stride_d, int stride_h, int stride_w,
                                  int padding_d, int padding_h, int padding_w) {
    auto in_sizes = input.sizes();

    int batch_size = in_sizes[0];
    int channels = in_sizes[1];
    int in_d = in_sizes[2];
    int in_h = in_sizes[3];
    int in_w = in_sizes[4];

    int out_d = ( (in_d + 2 * padding_d - kernel_d) / stride_d ) + 1;
    int out_h = ( (in_h + 2 * padding_h - kernel_h) / stride_h ) + 1;
    int out_w = ( (in_w + 2 * padding_w - kernel_w) / stride_w ) + 1;

    auto output = torch::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    int total_elements = batch_size * channels * out_d * out_h * out_w;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    avgpool3d_gelu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(),
                                               batch_size, channels, in_d, in_h, in_w,
                                               out_d, out_h, out_w,
                                               kernel_d, kernel_h, kernel_w,
                                               stride_d, stride_h, stride_w,
                                               padding_d, padding_h, padding_w);

    return output;
}
"""

avgpool3d_gelu_cpp_source = """
torch::Tensor avgpool3d_gelu_cuda(torch::Tensor input,
                                  int kernel_d, int kernel_h, int kernel_w,
                                  int stride_d, int stride_h, int stride_w,
                                  int padding_d, int padding_h, int padding_w);
"""

# Compile the inline CUDA code
avgpool3d_gelu = load_inline(
    name='avgpool3d_gelu',
    cpp_sources=avgpool3d_gelu_cpp_source,
    cuda_sources=avgpool3d_gelu_source,
    functions=['avgpool3d_gelu_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a sum, layer normalization,
    and a fused average pooling and GELU activation using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avgpool3d_gelu = avgpool3d_gelu
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        x = self.norm(x)
        # Fused AvgPool3d and GELU activation
        x = self.avgpool3d_gelu.avgpool3d_gelu_cuda(
            x,
            self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2],  # kernel sizes
            self.pool_kernel_size[0], self.pool_kernel_size[1], self.pool_kernel_size[2],  # strides
            0, 0, 0  # paddings
        )
        return x