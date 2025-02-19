import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void maxpool1d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * channels * output_length) {
        int ow = idx % output_length;
        int c = (idx / output_length) % channels;
        int b = idx / (channels * output_length);
        
        float maxval = -1e10f;
        int start = ow * stride - padding;
        
        for (int k = 0; k < kernel_size; k++) {
            int iw = start + k * dilation;
            if (iw >= 0 && iw < input_length) {
                float val = input[b * channels * input_length + c * input_length + iw];
                maxval = max(maxval, val);
            }
        }
        
        output[idx] = maxval;
    }
}

torch::Tensor maxpool1d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);
    
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    
    auto output = torch::empty({batch_size, channels, output_length}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * channels * output_length + threads - 1) / threads;
    
    maxpool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

maxpool1d_cpp_source = """
torch::Tensor maxpool1d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride, 
    int padding,
    int dilation);
"""

maxpool1d_cuda = load_inline(
    name='maxpool1d_cuda',
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_cuda_source,
    functions=['maxpool1d_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        if return_indices:
            raise NotImplementedError("return_indices=True is not supported in custom CUDA implementation")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool1d_cuda.maxpool1d_cuda(
            x.cuda(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )