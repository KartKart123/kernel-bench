import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batched_matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);

    const int batch_size = A.size(0);
    const int m = A.size(1);
    const int k = A.size(2);
    const int n = B.size(2);

    auto C = torch::zeros({batch_size, m, n}, A.options());
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    for(int b = 0; b < batch_size; b++) {
        cublasSgemm(handle, 
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    B.data_ptr<float>() + b * k * n,
                    n,
                    A.data_ptr<float>() + b * m * k,
                    k,
                    &beta,
                    C.data_ptr<float>() + b * m * n,
                    n);
    }
    
    cublasDestroy(handle);
    return C;
}
"""

batched_matmul_cpp_source = """
torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

batched_matmul = load_inline(
    name='batched_matmul',
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_cuda_source,
    functions=['batched_matmul_cuda'],
    extra_cuda_cflags=['-lcublas'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.batched_matmul.batched_matmul_cuda(A, B)