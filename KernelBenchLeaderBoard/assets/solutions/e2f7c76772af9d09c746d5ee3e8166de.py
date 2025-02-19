import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rnn_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rnn_forward_kernel(
    const float* input,
    const float* hidden,
    const float* i2h_weight,
    const float* i2h_bias, 
    const float* h2o_weight,
    const float* h2o_bias,
    float* new_hidden,
    float* output,
    int batch_size,
    int input_size,
    int hidden_size,
    int output_size
) {
    // Each thread handles one element of the batch
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Step 1: Concatenate input and hidden (implicitly by accessing correct indices)
        
        // Step 2: Input to hidden transformation
        for (int h = 0; h < hidden_size; h++) {
            float sum = i2h_bias[h];
            
            // Input portion
            for (int i = 0; i < input_size; i++) {
                sum += input[batch_idx * input_size + i] * 
                       i2h_weight[h * (input_size + hidden_size) + i];
            }
            
            // Hidden portion 
            for (int i = 0; i < hidden_size; i++) {
                sum += hidden[batch_idx * hidden_size + i] * 
                       i2h_weight[h * (input_size + hidden_size) + input_size + i];
            }
            
            // Apply tanh activation
            new_hidden[batch_idx * hidden_size + h] = tanhf(sum);
        }
        
        // Step 3: Hidden to output transformation
        for (int o = 0; o < output_size; o++) {
            float sum = h2o_bias[o];
            for (int h = 0; h < hidden_size; h++) {
                sum += new_hidden[batch_idx * hidden_size + h] * 
                       h2o_weight[o * hidden_size + h];
            }
            output[batch_idx * output_size + o] = sum;
        }
    }
}

std::vector<torch::Tensor> rnn_forward_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = hidden.size(1);
    const int output_size = h2o_weight.size(0);

    auto new_hidden = torch::empty({batch_size, hidden_size}, input.options());
    auto output = torch::empty({batch_size, output_size}, input.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    rnn_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        hidden.data_ptr<float>(),
        i2h_weight.data_ptr<float>(),
        i2h_bias.data_ptr<float>(),
        h2o_weight.data_ptr<float>(),
        h2o_bias.data_ptr<float>(),
        new_hidden.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        output_size
    );

    return {output, new_hidden};
}
"""

rnn_cpp_source = """
std::vector<torch::Tensor> rnn_forward_cuda(
    torch::Tensor input,
    torch::Tensor hidden,
    torch::Tensor i2h_weight,
    torch::Tensor i2h_bias,
    torch::Tensor h2o_weight,
    torch::Tensor h2o_bias
);
"""

rnn_cuda = load_inline(
    name='rnn_cuda',
    cpp_sources=rnn_cpp_source,
    cuda_sources=rnn_cuda_source,
    functions=['rnn_forward_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.rnn_cuda = rnn_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden = self.hidden.to(x.device)
        output, self.hidden = self.rnn_cuda.rnn_forward_cuda(
            x,
            self.hidden,
            self.i2h.weight,
            self.i2h.bias,
            self.h2o.weight,
            self.h2o.bias
        )
        return output