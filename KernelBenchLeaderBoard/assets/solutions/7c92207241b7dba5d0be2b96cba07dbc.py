import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernels
conv_proj_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_proj_kernel(const float* input, const float* conv_weight, 
                               const float* proj_weight, float* output,
                               int batch_size, int in_channels, int height, int width,
                               int embed_dim, int patch_size) {
    
    int b = blockIdx.x;
    int e = blockIdx.y;
    
    if (b < batch_size && e < embed_dim) {
        float sum = 0.0f;
        
        // Convolution + Flattening + Linear projection fused
        for(int h = 0; h < height/patch_size; h++) {
            for(int w = 0; w < width/patch_size; w++) {
                for(int ph = 0; ph < patch_size; ph++) {
                    for(int pw = 0; pw < patch_size; pw++) {
                        for(int c = 0; c < in_channels; c++) {
                            int in_idx = b * (in_channels * height * width) +
                                       c * (height * width) +
                                       (h*patch_size + ph) * width +
                                       (w*patch_size + pw);
                                       
                            int weight_idx = e * (in_channels * patch_size * patch_size) +
                                           c * (patch_size * patch_size) +
                                           ph * patch_size + pw;
                                           
                            sum += input[in_idx] * conv_weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        output[b * embed_dim + e] = sum;
    }
}

std::vector<torch::Tensor> conv_proj_cuda(torch::Tensor input, 
                                        torch::Tensor conv_weight,
                                        torch::Tensor proj_weight) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1); 
    const int height = input.size(2);
    const int width = input.size(3);
    const int embed_dim = conv_weight.size(0);
    const int patch_size = conv_weight.size(2);
    
    auto output = torch::zeros({batch_size, embed_dim}, input.options());
    
    dim3 threads(32, 32);
    dim3 blocks((batch_size + threads.x - 1) / threads.x,
                (embed_dim + threads.y - 1) / threads.y);
                
    conv_proj_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        proj_weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, height, width,
        embed_dim, patch_size);
        
    return {output};
}
"""

conv_proj_cpp_source = """
std::vector<torch::Tensor> conv_proj_cuda(
    torch::Tensor input,
    torch::Tensor conv_weight, 
    torch::Tensor proj_weight);
"""

# Compile custom kernels
conv_proj = load_inline(
    name='conv_proj',
    cpp_sources=conv_proj_cpp_source,
    cuda_sources=conv_proj_source,
    functions=['conv_proj_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, patch_size=4, in_channels=3):
        super(ModelNew, self).__init__()
        
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.linear_proj = nn.Linear(embed_dim * (32 // patch_size) * (32 // patch_size), embed_dim)
        
        self.conv_proj = conv_proj
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                     dim_feedforward=int(embed_dim * mlp_ratio), dropout=0.0)
            for _ in range(num_layers)
        ])
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        
        # Fused conv + projection
        x = self.conv_proj.conv_proj_cuda(x, self.conv1.weight, self.linear_proj.weight)[0]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)

        # Transformer layers  
        for layer in self.transformer_layers:
            x = layer(x)

        x = x[:, 0]
        x = self.fc_out(x)
        
        return x