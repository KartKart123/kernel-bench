import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for fused softmax + sigmoid
__global__ void fused_softmax_sigmoid_kernel(float* input, float* output, 
    int batch_size, int channels, int depth, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = depth * height * width;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int b = idx / (channels * spatial_size);
        int rem = idx % (channels * spatial_size);
        int c = rem / spatial_size;
        int s = rem % spatial_size;
        
        // Calculate softmax denominator for this spatial location
        float sum = 0.0f;
        float max_val = -INFINITY;
        
        for(int ch = 0; ch < channels; ch++) {
            int offset = b * channels * spatial_size + ch * spatial_size + s;
            max_val = max(max_val, input[offset]);
        }
        
        for(int ch = 0; ch < channels; ch++) {
            int offset = b * channels * spatial_size + ch * spatial_size + s;
            sum += expf(input[offset] - max_val);
        }
        
        // Calculate softmax and apply sigmoid
        int curr_idx = b * channels * spatial_size + c * spatial_size + s;
        float softmax_val = expf(input[curr_idx] - max_val) / sum;
        output[curr_idx] = 1.0f / (1.0f + expf(-softmax_val));
    }
}

torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    const int total_elements = batch_size * channels * depth * height * width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_softmax_sigmoid_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, depth, height, width
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_softmax_sigmoid_cuda(torch::Tensor input);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_softmax_sigmoid_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=padding, 
                                               output_padding=output_padding, bias=bias)
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        return self.fused_ops.fused_softmax_sigmoid_cuda(x)