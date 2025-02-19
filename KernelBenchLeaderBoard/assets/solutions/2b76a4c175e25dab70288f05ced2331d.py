import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused GEMM + BatchNorm + GELU kernel
__global__ void fused_gemm_bn_gelu_kernel(
    const float* input,
    const float* weight,
    const float* bias, 
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_mean,
    const float* bn_var,
    float* output,
    int batch_size,
    int in_features,
    int out_features) {
    
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        
        // GEMM
        for (int k = 0; k < in_features; k++) {
            sum += input[row * in_features + k] * weight[col * in_features + k];
        }
        sum += bias[col];
        
        // BatchNorm
        float bn_output = (sum - bn_mean[col]) / sqrt(bn_var[col] + 1e-5);
        bn_output = bn_output * bn_weight[col] + bn_bias[col];
        
        // GELU
        output[row * out_features + col] = bn_output * 0.5f * (1.0f + tanhf(0.797885f * bn_output + 0.035677f * bn_output * bn_output * bn_output));
    }
}

// Fused GroupNorm + Mean + ReLU kernel
__global__ void fused_gn_mean_relu_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int channels,
    int num_groups) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int group_size = channels / num_groups;
    
    if (tid < batch_size) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        
        // Calculate mean and variance for GroupNorm
        for (int c = 0; c < channels; c++) {
            float val = input[tid * channels + c];
            sum += val;
            sq_sum += val * val;
        }
        
        float mean = sum / channels;
        float var = (sq_sum / channels) - (mean * mean);
        float std = sqrt(var + 1e-5);
        
        // Apply GroupNorm + Mean + ReLU
        float group_sum = 0.0f;
        for (int c = 0; c < channels; c++) {
            int group_idx = c / group_size;
            float normalized = (input[tid * channels + c] - mean) / std;
            float gn_out = normalized * gamma[c] + beta[c];
            group_sum += gn_out;
        }
        
        float final_out = group_sum / channels;
        output[tid] = final_out > 0 ? final_out : 0;
    }
}

std::vector<torch::Tensor> fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups) {
    
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto intermediate = torch::zeros({batch_size, out_features}, input.options());
    auto output = torch::zeros({batch_size, 1}, input.options());
    
    const int threads = 256;
    const dim3 blocks1((batch_size + threads - 1) / threads, (out_features + threads - 1) / threads);
    
    fused_gemm_bn_gelu_kernel<<<blocks1, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(),
        bn_var.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    const int blocks2 = (batch_size + threads - 1) / threads;
    
    fused_gn_mean_relu_kernel<<<blocks2, threads>>>(
        intermediate.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_features,
        num_groups
    );
    
    return {output};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int num_groups);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('bn_mean', torch.zeros(out_features))
        self.register_buffer('bn_var', torch.ones(out_features))
        
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))
        
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(
            x,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_mean,
            self.bn_var,
            self.gn_weight,
            self.gn_bias,
            self.num_groups
        )[0]