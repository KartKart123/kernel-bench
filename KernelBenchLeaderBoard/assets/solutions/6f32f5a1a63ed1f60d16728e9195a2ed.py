import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for patch embedding
patch_embedding_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void patch_embedding_kernel(const float* img, float* out, int batch_size, int num_patches, int patch_dim, int patch_size, int channels, int image_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_patches) {
        int batch_idx = idx / num_patches;
        int patch_idx = idx % num_patches;
        int patch_row = (patch_idx / (image_size / patch_size)) * patch_size;
        int patch_col = (patch_idx % (image_size / patch_size)) * patch_size;

        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < patch_size; ++i) {
                for (int j = 0; j < patch_size; ++j) {
                    int img_idx = ((batch_idx * channels + c) * image_size + (patch_row + i)) * image_size + (patch_col + j);
                    int out_idx = ((batch_idx * num_patches + patch_idx) * patch_dim) + (c * patch_size * patch_size + i * patch_size + j);
                    out[out_idx] = img[img_idx];
                }
            }
        }
    }
}

torch::Tensor patch_embedding_cuda(torch::Tensor img, int patch_size, int channels, int image_size) {
    int batch_size = img.size(0);
    int num_patches = (image_size / patch_size) * (image_size / patch_size);
    int patch_dim = channels * patch_size * patch_size;
    auto out = torch::empty({batch_size, num_patches, patch_dim}, img.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * num_patches + block_size - 1) / block_size;

    patch_embedding_kernel<<<num_blocks, block_size>>>(
        img.data_ptr<float>(), out.data_ptr<float>(), batch_size, num_patches, patch_dim, patch_size, channels, image_size
    );

    return out;
}
"""

patch_embedding_cpp_source = "torch::Tensor patch_embedding_cuda(torch::Tensor img, int patch_size, int channels, int image_size);"

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
        
        x = self.patch_embedding.patch_embedding_cuda(img, p, img.shape[1], img.shape[2])
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)