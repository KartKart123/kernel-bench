import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MaxPool2d
maxpool2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace at;

__global__ void maxpool2d_cuda_kernel(const float* __restrict__ input, float* __restrict__ output,
                                      int N, int C, int H_in, int W_in,
                                      int H_out, int W_out, int kernel_size,
                                      int stride, int padding) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    
    if (index >= total) return;
    
    int w_out = index % W_out;
    int h_out = (index / W_out) % H_out;
    int c = (index / (W_out * H_out)) % C;
    int n = index / (C * H_out * W_out);
    
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    int h_end = min(h_start + kernel_size, H_in + padding);
    int w_end = min(w_start + kernel_size, W_in + padding);
    h_start = max(h_start, 0);
    w_start = max(w_start, 0);
    h_end = min(h_end, H_in);
    w_end = min(w_end, W_in);
    
    float max_val = -FLT_MAX;
    for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
            int input_idx = n * C * H_in * W_in + c * H_in * W_in + h * W_in + w;
            float val = input[input_idx];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    int output_idx = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
    output[output_idx] = max_val;
}

torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({N, C, H_out, W_out}, options);

    int total = N * C * H_out * W_out;
    int threads = 1024;
    int blocks = (total + threads - 1) / threads;

    maxpool2d_cuda_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out,
        kernel_size, stride, padding);

    return output;
}
"""

maxpool2d_cpp_source = """
torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

# Compile the inline CUDA code for MaxPool2d
maxpool2d = load_inline(
    name='maxpool2d',
    cpp_sources=maxpool2d_cpp_source,
    cuda_sources=maxpool2d_source,
    functions=['maxpool2d_cuda'],
    verbose=False
)

class MaxPool2dCUDA(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool2dCUDA, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return maxpool2d.maxpool2d_cuda(x, self.kernel_size, self.stride, self.padding)

class InceptionModuleNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModuleNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            MaxPool2dCUDA(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = MaxPool2dCUDA(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = MaxPool2dCUDA(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = InceptionModuleNew(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleNew(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = MaxPool2dCUDA(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModuleNew(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleNew(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleNew(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleNew(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleNew(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = MaxPool2dCUDA(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionModuleNew(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleNew(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x