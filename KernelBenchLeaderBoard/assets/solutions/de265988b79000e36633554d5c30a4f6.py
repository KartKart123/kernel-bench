import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for Softmax
softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cfloat>

__global__ void softmax_kernel_batch(const float* __restrict__ input, float* __restrict__ output, int batch_size, int dim) {
    // Each block handles one batch
    int batch = blockIdx.x;
    if (batch < batch_size) {
        // Shared memory for max and sum
        extern __shared__ float sdata[];
        float* smax = sdata;
        float* ssum = sdata + blockDim.x;

        // Each thread handles index tid
        int tid = threadIdx.x;
        float max_val = -FLT_MAX;
        for (int i = tid; i < dim; i += blockDim.x) {
            float val = input[batch * dim + i];
            if (val > max_val)
                max_val = val;
        }
        // Reduce max_val across threads
        smax[threadIdx.x] = max_val;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
            if (threadIdx.x < s) {
                if (smax[threadIdx.x] < smax[threadIdx.x + s]) {
                    smax[threadIdx.x] = smax[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        max_val = smax[0];
        __syncthreads();

        // Now compute the sum of exp(x - max)
        float sum = 0.0f;
        for (int i = tid; i < dim; i += blockDim.x) {
            float val = __expf(input[batch * dim + i] - max_val);
            output[batch * dim + i] = val; // Temporarily store exp(x - max)
            sum += val;
        }
        // Reduce sum across threads
        ssum[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
            if (threadIdx.x < s) {
                ssum[threadIdx.x] += ssum[threadIdx.x + s];
            }
            __syncthreads();
        }
        sum = ssum[0];
        __syncthreads();

        // Finally compute output = exp(x - max) / sum
        for (int i = tid; i < dim; i += blockDim.x) {
            output[batch * dim + i] = output[batch * dim + i] / sum;
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    // Ensure input is contiguous and on CUDA
    auto input_contiguous = input.contiguous();
    auto batch_size = input_contiguous.size(0);
    auto dim = input_contiguous.size(1);
    auto output = torch::empty_like(input_contiguous);

    int threads = 512; // Adjust as needed
    int blocks = batch_size;

    size_t shared_mem_size = 2 * threads * sizeof(float);

    softmax_kernel_batch<<<blocks, threads, shared_mem_size>>>(input_contiguous.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

softmax_cpp_source = """
#include <torch/extension.h>
torch::Tensor softmax_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code for Softmax
softmax_module = load_inline(
    name='softmax_cuda',
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_cuda_source,
    functions=['softmax_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_module.softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda(x)