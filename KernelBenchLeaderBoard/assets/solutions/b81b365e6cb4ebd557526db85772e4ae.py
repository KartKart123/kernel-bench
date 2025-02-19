import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batchnorm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void compute_mean_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ mean,
    int N,
    int C,
    int H,
    int W)
{
    int c = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ scalar_t shared_sum[BLOCK_SIZE];
    scalar_t sum = 0;

    int total_elements = N * H * W;
    for (int idx = tid; idx < total_elements; idx += blockDim.x)
    {
        int n = idx / (H * W);
        int hw = idx % (H * W);
        int h = hw / W;
        int w = hw % W;
        scalar_t val = x[((n * C + c) * H + h) * W + w];
        sum += val;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduce sum within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        mean[c] = shared_sum[0] / total_elements;
    }
}

template <typename scalar_t>
__global__ void compute_variance_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    scalar_t* __restrict__ variance,
    int N,
    int C,
    int H,
    int W)
{
    int c = blockIdx.x;
    int tid = threadIdx.x;
    scalar_t m = mean[c];

    __shared__ scalar_t shared_sum[BLOCK_SIZE];
    scalar_t sum = 0;

    int total_elements = N * H * W;
    for (int idx = tid; idx < total_elements; idx += blockDim.x)
    {
        int n = idx / (H * W);
        int hw = idx % (H * W);
        int h = hw / W;
        int w = hw % W;
        scalar_t val = x[((n * C + c) * H + h) * W + w];
        sum += (val - m) * (val - m);
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduce sum within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        variance[c] = shared_sum[0] / total_elements;
    }
}

template <typename scalar_t>
__global__ void batchnorm_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ variance,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t epsilon,
    int N,
    int C,
    int H,
    int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;

    if (idx < total_elements)
    {
        int n = idx / (C * H * W);
        int c = (idx / (H * W)) % C;
        int h = (idx / W) % H;
        int w = idx % W;

        scalar_t m = mean[c];
        scalar_t var = variance[c];
        scalar_t g = gamma ? gamma[c] : static_cast<scalar_t>(1.0);
        scalar_t b = beta ? beta[c] : static_cast<scalar_t>(0.0);

        scalar_t val = x[((n * C + c) * H + h) * W + w];
        val = (val - m) / sqrtf(var + epsilon);
        val = val * g + b;
        y[((n * C + c) * H + h) * W + w] = val;
    }
}

torch::Tensor batchnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    bool training,
    double momentum,
    double epsilon)
{
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    auto y = torch::empty_like(x);

    auto mean = torch::zeros({C}, x.options());
    auto variance = torch::zeros({C}, x.options());

    const int threads = BLOCK_SIZE;
    const int blocks = C;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "compute_mean_cuda", ([&] {
        compute_mean_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            N, C, H, W);
    }));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "compute_variance_cuda", ([&] {
        compute_variance_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            variance.data_ptr<scalar_t>(),
            N, C, H, W);
    }));

    if (training)
    {
        running_mean.mul_(1 - momentum).add_(mean * momentum);
        running_var.mul_(1 - momentum).add_(variance * momentum);
    }
    else
    {
        mean.copy_(running_mean);
        variance.copy_(running_var);
    }

    const int total_threads = 1024;
    const int total_blocks = (N * C * H * W + total_threads - 1) / total_threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "batchnorm_forward_cuda", ([&] {
        batchnorm_forward_kernel<scalar_t><<<total_blocks, total_threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            variance.data_ptr<scalar_t>(),
            gamma.defined() ? gamma.data_ptr<scalar_t>() : nullptr,
            beta.defined() ? beta.data_ptr<scalar_t>() : nullptr,
            (scalar_t)epsilon,
            N, C, H, W);
    }));

    return y;
}
"""

batchnorm_cpp_source = """
torch::Tensor batchnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    bool training,
    double momentum,
    double epsilon);
"""

batchnorm = load_inline(
    name='batchnorm',
    cpp_sources=batchnorm_cpp_source,
    cuda_sources=batchnorm_cuda_source,
    functions=['batchnorm_forward_cuda'],
    verbose=False,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3'],
)

class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, device='cuda'))
            self.bias = nn.Parameter(torch.zeros(num_features, device='cuda'))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, device='cuda'))
            self.register_buffer('running_var', torch.ones(num_features, device='cuda'))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        y = batchnorm.batchnorm_forward_cuda(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps
        )
        return y

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super(ModelNew, self).__init__()
        self.bn = CustomBatchNorm2d(num_features=num_features)

    def forward(self, x):
        return self.bn(x)