import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernels
patch_embedding_cuda = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void unfold_and_embed_kernel(
    const float* input, const float* weights, const float* bias,
    float* output, int batch_size, int channels, int height, int width,
    int patch_size, int dim, int num_patches) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_patches * dim) return;
    
    int b = idx / (num_patches * dim);
    int p = (idx % (num_patches * dim)) / dim;
    int d = idx % dim;
    
    int ph = p / (width/patch_size);
    int pw = p % (width/patch_size);
    
    float sum = 0.0f;
    for(int c = 0; c < channels; c++) {
        for(int i = 0; i < patch_size; i++) {
            for(int j = 0; j < patch_size; j++) {
                int in_idx = b * channels * height * width +
                            c * height * width +
                            (ph * patch_size + i) * width +
                            (pw * patch_size + j);
                int w_idx = (c * patch_size * patch_size + i * patch_size + j) * dim + d;
                sum += input[in_idx] * weights[w_idx];
            }
        }
    }
    output[idx] = sum + bias[d];
}

torch::Tensor patch_embed_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    int patch_size,
    int dim) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int num_patches = (height/patch_size) * (width/patch_size);
    
    auto output = torch::empty({batch_size, num_patches, dim}, 
                             input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * num_patches * dim + threads - 1) / threads;
    
    unfold_and_embed_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width,
        patch_size, dim, num_patches);
    
    return output;
}
"""

patch_embedding_cpp = """
torch::Tensor patch_embed_cuda(
    torch::Tensor input,
    torch::Tensor weights, 
    torch::Tensor bias,
    int patch_size,
    int dim);
"""

patch_embed = load_inline(
    name='patch_embed',
    cpp_sources=patch_embedding_cpp,
    cuda_sources=patch_embedding_cuda,
    functions=['patch_embed_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        
        self.patch_embed = patch_embed
    
    def forward(self, img):
        # Use custom CUDA kernel for patch embedding
        x = self.patch_embed.patch_embed_cuda(
            img,
            self.patch_to_embedding.weight,
            self.patch_to_embedding.bias,
            self.patch_size,
            self.dim
        )
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)