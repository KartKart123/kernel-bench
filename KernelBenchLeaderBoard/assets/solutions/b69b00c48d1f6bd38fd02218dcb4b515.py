import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triplet_margin_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void triplet_margin_loss_kernel(const float* anchor, const float* positive, const float* negative, float margin, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float pos_dist = 0.0f;
        float neg_dist = 0.0f;
        for (int i = 0; i < 4096; i++) {
            float tmp = anchor[idx * 4096 + i] - positive[idx * 4096 + i];
            pos_dist += tmp * tmp;
            tmp = anchor[idx * 4096 + i] - negative[idx * 4096 + i];
            neg_dist += tmp * tmp;
        }
        pos_dist = sqrtf(pos_dist);
        neg_dist = sqrtf(neg_dist);
        out[idx] = max(0.0f, margin + pos_dist - neg_dist);
    }
}

torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin) {
    auto size = anchor.size(0);
    auto out = torch::zeros(size, torch::TensorOptions().device(torch::kCUDA));

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    triplet_margin_loss_kernel<<<num_blocks, block_size>>>(anchor.data_ptr<float>(), positive.data_ptr<float>(), negative.data_ptr<float>(), margin, out.data_ptr<float>(), size);

    return out.mean();
}
"""

triplet_margin_loss_cpp_source = (
    "torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);"
)

triplet_margin_loss = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_margin_loss = triplet_margin_loss

    def forward(self, anchor, positive, negative):
        return self.triplet_margin_loss.triplet_margin_loss_cuda(anchor, positive, negative, self.margin)