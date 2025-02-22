import re
from typing import List

def has_correct_block_structure(content: str):
    format_pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    parse_match = re.match(format_pattern, content, re.DOTALL | re.MULTILINE)
    return bool(parse_match)

def has_exactly_one_block_of_code(content: str) -> bool:
    code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", content, re.DOTALL)
    return len(code_blocks) == 1

def parse_content(content) -> List[str]:
    assert has_correct_block_structure(content)
    assert has_exactly_one_block_of_code(content)

    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    thought_process = think_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    verbal_response = answer_match.group(1).strip()
    
    code_block_match = re.search(r"```(?:python)?\n(.*?)\n```", content, re.DOTALL)
    code_block = code_block_match.group(1).strip()
    code_block.remove_prefix("python")

    assert(all(a is not None for a in (thought_process, verbal_response, code_block)))
    return [thought_process, verbal_response, code_block]

def code_block_contains_cuda_kernel(code_block) -> bool:
    must_contains = ["__global__", "load_inline(", "#include <torch/extension.h"]
    if any((s not in code_block for s in must_contains)):
        #TODO: more elaborate parsing.
        return False
    return True

def test():
    response = "<think>\nAlright, I need to help the user replace PyTorch's `torch.cumprod` with a custom CUDA kernel optimized for the NVIDIA H100 GPU. Let me break down the steps I took to arrive at the solution.\n\nFirst, I considered the architecture provided. The original model uses `torch.cumprod`, which is a high-level PyTorch function. My goal is to replace this with a custom CUDA kernel for better performance on the H100.\n\nI started by understanding how `torch.cumprod` works. It computes the cumulative product along a specified dimension. To replicate this behavior with a CUDA kernel, I needed to implement the cumulative product algorithm manually.\n\nNext, I thought about how to structure the CUDA kernel. The cumulative product can be computed in-place, updating each element based on the previous product. I decided to use shared memory within the kernel to store intermediate results, as shared memory is faster than global memory on CUDA.\n\nI considered the thread hierarchy. Each thread block would handle a segment of the input tensor, and each thread within the block would compute a portion of the cumulative product. I made sure to handle the first element separately since it doesn't have a previous value.\n\nI also thought about the grid and block configuration. The number of blocks and threads per block needed to be chosen to maximize occupancy and minimize memory access latencies. I chose a block size of 256, which is a common choice for kernels on the H100 due to its warp size of 32, allowing for 8 warps per block.\n\nI then wrote the CUDA kernel source code, ensuring that it correctly computes the cumulative product. After writing the kernel, I used `torch.utils.cpp_extension.load_inline` to compile the kernel into a custom CUDA operator in PyTorch. This allows the kernel to be called directly from the forward pass of the model.\n\nFinally, I integrated the custom kernel into the new model architecture, `ModelNew`, replacing the `torch.cumprod` call with the new custom operation. I made sure to test the code to ensure it works correctly and efficiently on the H100 GPU.\n\nThroughout this process, I focused on ensuring correctness, optimizing for speed, and making sure the custom kernel leverages the H100's capabilities effectively.\n</think>\n\n```\nimport torch\nimport torch.nn as nn\nfrom torch.utils.cpp_extension import load_inline\n\n# Define the custom CUDA kernel for cumulative product\ncumulative_product_source = \"\"\"\n#include <torch/extension.h>\n#include <cuda_runtime.h>\n\n__global__ void cumulative_product_kernel(const float* input, const int64_t* dim, int64_t* size, float* output) {\n    int pos = blockIdx.x * blockDim.x + threadIdx.x;\n    if (pos < *size) {\n        if (pos == 0) {\n            output[pos] = input[pos];\n        } else {\n            int pos_dim = pos;\n            for (int d = 1; d < *dim; d++) {\n                pos_dim += (pos_dim >> (8 * (d - 1))) & mask;\n            }\n            output[pos] = output[pos_dim] * input[pos];\n        }\n    }\n}\n\n// A helper function to compute cumulative product along a dimension\ntorch::Tensor cumulative_product_cuda(torch::Tensor input, int64_t dim) {\n    int64_t n = input.numel();\n    int64_t size[2];\n    size[0] = input.dim(dim);\n    size[1] = n / size[0];\n    \n    float* output = new float[n];\n    \n    int mask = 0;\n    for (int d = 1; d < 2; d++) {\n        mask |= (1 << (7 - d));\n    }\n    \n    const int block_size = 256;\n    const int num_blocks = (n + block_size - 1) / block_size;\n    \n    cumulative_product_kernel<<<num_blocks, block_size>>>(\n        input.data_ptr<float>(), \n        dim.data(), \n        size, \n        output\n    );\n    \n    return torch::Tensor(output);\n}\n\"\"\"\n\n# Compile the custom CUDA kernel\ncumulative_product = load_inline(\n    name=\"cumulative_product\",\n    cpp_sources=\"\",\n    cuda_sources=cumulative_product_source,\n    functions=[\"cumulative_product_cuda\"],\n    verbose=True,\n    extra_cflags=[\"\"],\n    extra_ldflags=[\"\"],\n    include_paths=[\"/usr/local/cudainclude\"]\n)\n\nclass ModelNew(nn.Module):\n    def __init__(self, dim):\n        super(ModelNew, self).__init__()\n        self.dim = dim\n        \n    def forward(self, x):\n        return cumulative_product.cumulative_product_forward(x, self.dim)\n```\n\nThis implementation:\n1. Uses CUDA to accelerate the cumulative product computation\n2. Leverages shared memory for faster access within threads\n3. Uses block and thread parallelism to maximize GPU occupancy\n4. Handles the dimension parameter like the original PyTorch implementation\n5. Is optimized for the H100 GPU architecture\n\nThe custom kernel replaces the standard `torch.cumprod` operation while maintaining the same functionality and interface. The performance improvement should be significant on the H100 due to the parallel nature of the CUDA kernel and its efficient use of shared memory.\n\nTo use this implementation, simply instantiate the `ModelNew` class with your desired dimension and pass your tensor through the forward pass."
    print(has_correct_block_structure(response))
    print(has_exactly_one_block_of_code(response))

if __name__ == '__main__':
    test()
