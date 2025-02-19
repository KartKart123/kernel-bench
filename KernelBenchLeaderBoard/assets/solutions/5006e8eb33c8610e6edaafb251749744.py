import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_relu_conv_avgpool_kernel(
    const float* input,
    const float* bn_weight,
    const float* bn_bias, 
    const float* bn_mean,
    const float* bn_var,
    const float* conv_weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels, 
    const int height,
    const int width,
    const float eps) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * out_channels * (height/2) * (width/2);
    
    if(tid < total_threads) {
        // Calculate output indices
        const int out_w = tid % (width/2);
        const int out_h = (tid / (width/2)) % (height/2);
        const int out_c = (tid / ((width/2) * (height/2))) % out_channels;
        const int out_n = tid / (out_channels * (height/2) * (width/2));

        float sum = 0.0f;
        
        // For each input channel
        for(int ic = 0; ic < in_channels; ic++) {
            // For each pixel in 2x2 pooling window
            for(int ph = 0; ph < 2; ph++) {
                for(int pw = 0; pw < 2; pw++) {
                    const int in_h = out_h * 2 + ph;
                    const int in_w = out_w * 2 + pw;
                    
                    // Get input value
                    const int in_idx = ((out_n * in_channels + ic) * height + in_h) * width + in_w;
                    float val = input[in_idx];
                    
                    // Apply BatchNorm
                    val = (val - bn_mean[ic]) / sqrt(bn_var[ic] + eps);
                    val = val * bn_weight[ic] + bn_bias[ic];
                    
                    // Apply ReLU
                    val = val > 0 ? val : 0;
                    
                    // Accumulate conv * pool
                    sum += val * conv_weight[out_c * in_channels + ic];
                }
            }
        }
        
        // Average pooling
        output[tid] = sum / 4.0f;
    }
}

std::vector<torch::Tensor> fused_transition_cuda(
    torch::Tensor input,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean, 
    torch::Tensor bn_var,
    torch::Tensor conv_weight) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = conv_weight.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height/2, width/2}, 
                             input.options());

    const int threads = 256;
    const int blocks = (batch_size * out_channels * (height/2) * (width/2) + threads - 1) / threads;

    fused_bn_relu_conv_avgpool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        1e-5f
    );

    return {output};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_transition_cuda(
    torch::Tensor input,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var, 
    torch::Tensor conv_weight);
"""

fused_ops = load_inline(
    name='fused_transition',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_transition_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, 
                            kernel_size=1, bias=False)
        self.fused_op = fused_ops.fused_transition_cuda

    def forward(self, x):
        return self.fused_op(x, 
                           self.bn.weight,
                           self.bn.bias,
                           self.bn.running_mean,
                           self.bn.running_var,
                           self.conv.weight)[0]