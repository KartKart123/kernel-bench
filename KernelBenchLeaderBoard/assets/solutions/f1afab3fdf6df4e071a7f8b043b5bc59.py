import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU + MaxPool2d
relu_maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void relu_maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * pooled_height * pooled_width;

    if (index < total_elements)
    {
        const int w_out = index % pooled_width;
        const int h_out = (index / pooled_width) % pooled_height;
        const int c = (index / (pooled_width * pooled_height)) % channels;
        const int n = index / (channels * pooled_height * pooled_width);

        const int h_in_base = h_out * 2;
        const int w_in_base = w_out * 2;

        float max_val = -FLT_MAX;
        for (int kh = 0; kh < 2; ++kh)
        {
            for (int kw = 0; kw < 2; ++kw)
            {
                int h_in = h_in_base + kh;
                int w_in = w_in_base + kw;

                if (h_in < height && w_in < width)
                {
                    int input_index = n * channels * height * width +
                                      c * height * width +
                                      h_in * width +
                                      w_in;
                    float val = input[input_index];
                    val = fmaxf(val, 0.0f); // ReLU activation
                    if (val > max_val)
                    {
                        max_val = val;
                    }
                }
            }
        }

        int output_index = n * channels * pooled_height * pooled_width +
                           c * pooled_height * pooled_width +
                           h_out * pooled_width +
                           w_out;
        output[output_index] = max_val;
    }
}

torch::Tensor relu_maxpool2d_cuda(torch::Tensor input)
{
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const int pooled_height = (height + 1) / 2;
    const int pooled_width = (width + 1) / 2;

    auto output = torch::empty({batch_size, channels, pooled_height, pooled_width}, input.options());

    const int threads = 1024;
    const int total_elements = batch_size * channels * pooled_height * pooled_width;
    const int blocks = (total_elements + threads - 1) / threads;

    relu_maxpool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        pooled_height,
        pooled_width
    );

    return output;
}
"""

relu_maxpool2d_cpp_source = "torch::Tensor relu_maxpool2d_cuda(torch::Tensor input);"

# Compile the inline CUDA code for ReLU + MaxPool2d
relu_maxpool2d = load_inline(
    name='relu_maxpool2d',
    cpp_sources=relu_maxpool2d_cpp_source,
    cuda_sources=relu_maxpool2d_source,
    functions=['relu_maxpool2d_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3'],
    extra_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        Optimized LeNet-5 architecture with custom CUDA operators.
        """
        super(ModelNew, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

        self.relu_maxpool2d = relu_maxpool2d

    def forward(self, x):
        """
        Forward pass of the optimized LeNet-5 model.
        """
        # First convolutional layer
        x = self.conv1(x)
        x = self.relu_maxpool2d.relu_maxpool2d_cuda(x)
        
        # Second convolutional layer
        x = self.conv2(x)
        x = self.relu_maxpool2d.relu_maxpool2d_cuda(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x