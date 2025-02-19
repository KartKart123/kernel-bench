import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool3d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels, 
    const int dim1,
    const int dim2,
    const int dim3,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int out_dim1,
    const int out_dim2,
    const int out_dim3
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3;
    
    if (idx < total_elements) {
        const int out_pos = idx;
        const int out_z = out_pos % out_dim3;
        const int out_y = (out_pos / out_dim3) % out_dim2;
        const int out_x = (out_pos / (out_dim3 * out_dim2)) % out_dim1;
        const int c = (out_pos / (out_dim3 * out_dim2 * out_dim1)) % channels;
        const int b = out_pos / (out_dim3 * out_dim2 * out_dim1 * channels);

        float maxval = -1e38;
        
        const int start_x = out_x * stride - padding;
        const int start_y = out_y * stride - padding;
        const int start_z = out_z * stride - padding;
        
        for(int kx = 0; kx < kernel_size; kx++) {
            const int in_x = start_x + kx * dilation;
            if (in_x >= 0 && in_x < dim1) {
                for(int ky = 0; ky < kernel_size; ky++) {
                    const int in_y = start_y + ky * dilation;
                    if (in_y >= 0 && in_y < dim2) {
                        for(int kz = 0; kz < kernel_size; kz++) {
                            const int in_z = start_z + kz * dilation;
                            if (in_z >= 0 && in_z < dim3) {
                                const int in_idx = ((b * channels + c) * dim1 + in_x) * dim2 * dim3 + 
                                                 in_y * dim3 + in_z;
                                maxval = max(maxval, input[in_idx]);
                            }
                        }
                    }
                }
            }
        }
        output[out_pos] = maxval;
    }
}

torch::Tensor maxpool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int dim1 = input.size(2);
    const int dim2 = input.size(3);
    const int dim3 = input.size(4);
    
    const int out_dim1 = (dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_dim2 = (dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_dim3 = (dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, channels, out_dim1, out_dim2, out_dim3}, 
                             input.options());

    const int total_elements = batch_size * channels * out_dim1 * out_dim2 * out_dim3;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        dim1,
        dim2,
        dim3,
        kernel_size,
        stride,
        padding, 
        dilation,
        out_dim1,
        out_dim2,
        out_dim3
    );

    return output;
}
"""

maxpool3d_cpp_source = """
torch::Tensor maxpool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride, 
    int padding,
    int dilation
);
"""

maxpool3d = load_inline(
    name='maxpool3d',
    cpp_sources=maxpool3d_cpp_source,
    cuda_sources=maxpool3d_cuda_source,
    functions=['maxpool3d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool3d.maxpool3d_cuda(x.cuda(), self.kernel_size, self.stride, 
                                      self.padding, self.dilation)