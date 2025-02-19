import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv_relu_maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_relu_maxpool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size, int stride, int pool_size) {

    int b = blockIdx.x;
    int oc = blockIdx.y;
    int oh = blockIdx.z / ((in_width-kernel_size+1)/2);
    int ow = blockIdx.z % ((in_width-kernel_size+1)/2);

    float maxval = -1e9;
    
    for(int ph = 0; ph < pool_size; ph++) {
        for(int pw = 0; pw < pool_size; pw++) {
            float conv_val = bias[oc];
            
            for(int ic = 0; ic < in_channels; ic++) {
                for(int kh = 0; kh < kernel_size; kh++) {
                    for(int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh*2 + ph + kh;
                        int iw = ow*2 + pw + kw;
                        
                        conv_val += input[b*in_channels*in_height*in_width + 
                                        ic*in_height*in_width +
                                        ih*in_width + iw] *
                                  weight[oc*in_channels*kernel_size*kernel_size +
                                        ic*kernel_size*kernel_size +
                                        kh*kernel_size + kw];
                    }
                }
            }
            
            float relu_val = conv_val > 0 ? conv_val : 0;
            maxval = max(maxval, relu_val);
        }
    }
    
    output[b*out_channels*((in_height-kernel_size+1)/2)*((in_width-kernel_size+1)/2) +
           oc*((in_height-kernel_size+1)/2)*((in_width-kernel_size+1)/2) +
           oh*((in_width-kernel_size+1)/2) + ow] = maxval;
}

torch::Tensor conv_relu_maxpool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1); 
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int pool_size = 2;
    
    const int out_height = (in_height - kernel_size + 1) / 2;
    const int out_width = (in_width - kernel_size + 1) / 2;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                             input.options());
    
    dim3 grid(batch_size, out_channels, out_height * out_width);
    dim3 block(1);
    
    conv_relu_maxpool_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size, 1, pool_size);
    
    return output;
}
"""

conv_relu_maxpool_cpp_source = """
torch::Tensor conv_relu_maxpool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_relu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int b = blockIdx.x;
    int o = blockIdx.y * blockDim.x + threadIdx.x;
    
    if(o < out_features) {
        float val = bias[o];
        for(int i = 0; i < in_features; i++) {
            val += input[b*in_features + i] * weight[o*in_features + i];
        }
        output[b*out_features + o] = val > 0 ? val : 0;
    }
}

torch::Tensor linear_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int threads = 256;
    const int blocks_y = (out_features + threads - 1) / threads;
    
    dim3 grid(batch_size, blocks_y);
    dim3 block(threads);
    
    linear_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features);
    
    return output;
}
"""

linear_relu_cpp_source = """
torch::Tensor linear_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

conv_relu_maxpool = load_inline(
    name='conv_relu_maxpool',
    cpp_sources=conv_relu_maxpool_cpp_source,
    cuda_sources=conv_relu_maxpool_source,
    functions=['conv_relu_maxpool_cuda'],
    verbose=True
)

linear_relu = load_inline(
    name='linear_relu',
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=['linear_relu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Fused conv+relu+maxpool operations
        x = conv_relu_maxpool.conv_relu_maxpool_cuda(
            x, self.conv1.weight, self.conv1.bias)
        x = conv_relu_maxpool.conv_relu_maxpool_cuda(
            x, self.conv2.weight, self.conv2.bias)
        
        x = x.view(-1, 16*5*5)
        
        # Fused linear+relu operations
        x = linear_relu.linear_relu_cuda(
            x, self.fc1.weight, self.fc1.bias)
        x = linear_relu.linear_relu_cuda(
            x, self.fc2.weight, self.fc2.bias)
        
        # Final linear layer (no ReLU)
        x = F.linear(x, self.fc3.weight, self.fc3.bias)
        
        return x