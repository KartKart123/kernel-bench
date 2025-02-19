import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for patch embedding
patch_embedding_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void patch_embedding_kernel(const float* img, float* out, int batch_size, int num_patches, int patch_dim, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_patches * dim) {
        int b = idx / (num_patches * dim);
        int p = (idx % (num_patches * dim)) / dim;
        int d = idx % dim;
        out[idx] = img[b * num_patches * patch_dim + p * patch_dim + d];
    }
}

torch::Tensor patch_embedding_cuda(torch::Tensor img, int num_patches, int patch_dim, int dim) {
    auto batch_size = img.size(0);
    auto out = torch::zeros({batch_size, num_patches, dim}, torch::dtype(torch::kFloat32).device(img.device()));

    const int block_size = 256;
    const int num_blocks = (batch_size * num_patches * dim + block_size - 1) / block_size;

    patch_embedding_kernel<<<num_blocks, block_size>>>(img.data_ptr<float>(), out.data_ptr<float>(), batch_size, num_patches, patch_dim, dim);

    return out;
}
"""

patch_embedding_cpp_source = "torch::Tensor patch_embedding_cuda(torch::Tensor img, int num_patches, int patch_dim, int dim);"

# Compile the inline CUDA code for patch embedding
patch_embedding = load_inline(
    name='patch_embedding',
    cpp_sources=patch_embedding_cpp_source,
    cuda_sources=patch_embedding_source,
    functions=['patch_embedding_cuda'],
    verbose=True,
    extra_cflags=[''],
    extra_ldflags=['']
)

class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
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
        
        self.patch_embedding = patch_embedding
    
    def forward(self, img):
        p = self.patch_size
        num_patches = (img.size(2) // p) * (img.size(3) // p)
        patch_dim = img.size(1) * p * p
        
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, patch_dim)
        x = self.patch_embedding.patch_embedding_cuda(x, num_patches, patch_dim, self.patch_to_embedding.in_features)
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)