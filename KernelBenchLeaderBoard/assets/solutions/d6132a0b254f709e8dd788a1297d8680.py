import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for ReLU operation
relu_cuda_source = """
#include <torch/extension.h>

__global__ void relu_forward(const float* __restrict__ input, float* __restrict__ output, const int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

torch::Tensor relu_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int num_elements = input.numel();
    const int threads = 1024;
    const int blocks = (num_elements + threads - 1) / threads;

    relu_forward<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_elements);

    return output;
}
"""

relu_cpp_source = "torch::Tensor relu_cuda_forward(torch::Tensor input);"

# Compile the CUDA code for ReLU operation
relu_cuda = load_inline(
    name='relu_cuda',
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_cuda_source,
    functions=['relu_cuda_forward'],
    verbose=True,
)

# Custom ReLU layer using the compiled CUDA kernel
class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()

    def forward(self, input):
        return relu_cuda.relu_cuda_forward(input)

class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(ModelNew, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            CustomReLU(),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )
    
    def forward(self, x):
        batch_size, _, height, width = x.size()
        total_output_channels = self.num_input_features + self.num_layers * self.growth_rate
        # Preallocate the output tensor to avoid repeated concatenations
        out = torch.empty(batch_size, total_output_channels, height, width, device=x.device, dtype=x.dtype)
        # Copy the input features to the beginning of the output tensor
        out[:, :self.num_input_features, :, :] = x
        current_channels = self.num_input_features
        input = x
        for idx, layer in enumerate(self.layers):
            new_features = layer(input)
            out[:, current_channels:current_channels + self.growth_rate, :, :] = new_features
            current_channels += self.growth_rate
            # Update input to include all features up to the current point
            input = out[:, :current_channels, :, :]
        return out