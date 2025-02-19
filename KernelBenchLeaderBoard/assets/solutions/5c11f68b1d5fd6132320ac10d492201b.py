import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_source = """
#include <torch/extension.h>

__device__ float smooth_l1_loss_kernel(float x) {
  float absx = fabsf(x);
  if (absx < 1.0f) {
    return 0.5f * absx * absx;
  } else {
    return absx - 0.5f;
  }
}

__global__ void smooth_l1_loss_kernel_launcher(const float* predictions, const float* targets, float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = smooth_l1_loss_kernel(predictions[i] - targets[i]);
  }
}


torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
  AT_ASSERTM(predictions.dim() == 1 || predictions.dim() == 2, "predictions must be 1D or 2D");
  AT_ASSERTM(targets.dim() == 1 || targets.dim() == 2, "targets must be 1D or 2D");
  AT_ASSERTM(predictions.sizes() == targets.sizes(), "predictions and targets must have the same size");
  AT_ASSERTM(predictions.scalar_type() == at::ScalarType::Float, "predictions must be float");
  AT_ASSERTM(targets.scalar_type() == at::ScalarType::Float, "targets must be float");

  auto size = predictions.numel();
  auto output = torch::zeros_like(predictions);

  const int block_size = 256;
  const int num_blocks = (size + block_size - 1) / block_size;

  smooth_l1_loss_kernel_launcher<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), output.data_ptr<float>(), size);

  return torch::mean(output);
}
"""

smooth_l1_loss_cpp_source = "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

smooth_l1_loss = load_inline(
    name='smooth_l1_loss',
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=['smooth_l1_loss_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.smooth_l1_loss.smooth_l1_loss_cuda(predictions, targets)