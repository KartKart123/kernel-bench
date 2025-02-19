import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

instance_norm_cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_mean_var_kernel(
    const float* __restrict__ x,
    float* mean,
    float* var,
    int N, int C, int H, int W) {
    int n = blockIdx.x;
    int c = blockIdx.y;
    int HW = H * W;

    extern __shared__ float shared_data[];
    float* s_sum = shared_data;
    float* s_sum2 = shared_data + blockDim.x;

    float sum = 0.0f;
    float sum2 = 0.0f;

    int thread_idx = threadIdx.x;
    int num_threads = blockDim.x;

    for (int i = thread_idx; i < HW; i += num_threads) {
        int index = ((n * C + c) * H * W) + i;
        float val = x[index];
        sum += val;
        sum2 += val * val;
    }

    s_sum[thread_idx] = sum;
    s_sum2[thread_idx] = sum2;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            s_sum[thread_idx] += s_sum[thread_idx + s];
            s_sum2[thread_idx] += s_sum2[thread_idx + s];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        float mean_val = s_sum[0] / HW;
        float var_val = s_sum2[0] / HW - mean_val * mean_val;
        mean[n * C + c] = mean_val;
        var[n * C + c] = var_val;
    }
}

__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* y,
    float epsilon,
    int N, int C, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    if (index >= total_elements) return;

    int n = index / (C * H * W);
    int c = (index / (H * W)) % C;

    float mean_val = mean[n * C + c];
    float var_val = var[n * C + c];

    float gamma_c = gamma[c];
    float beta_c = beta[c];

    float x_val = x[index];
    float y_val = gamma_c * (x_val - mean_val) / sqrtf(var_val + epsilon) + beta_c;
    y[index] = y_val;
}

torch::Tensor instance_norm_forward(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double epsilon) {
    x = x.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto y = torch::empty_like(x);

    auto mean = torch::empty({N, C}, x.options());
    auto var = torch::empty({N, C}, x.options());

    dim3 blockDim(256);
    dim3 gridDim(N, C);

    int shared_mem_size = 2 * blockDim.x * sizeof(float);

    compute_mean_var_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C, H, W
    );

    int total_elements = N * C * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    instance_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        (float)epsilon,
        N, C, H, W
    );

    return y;
}
'''

instance_norm_cpp_source = '''
torch::Tensor instance_norm_forward(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double epsilon);
'''

instance_norm = load_inline(
    name='instance_norm',
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_cuda_source,
    functions=['instance_norm_forward'],
    verbose=True,
)

class ModelNew(nn.Module):
    """
    Model with custom CUDA operator for Instance Normalization.
    """
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.epsilon = 1e-5
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_forward(x, self.gamma, self.beta, self.epsilon)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]