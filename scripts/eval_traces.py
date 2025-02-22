from pydra import Config
import os
import re
import pydra
import json
import random
from src.prompt_constructor import custom_prompt_generate_custom_cuda
from src.utils import read_file
from tqdm import tqdm
import torch
import multiprocessing as mp
from typing import List, Dict, Tuple
from functools import partial
from src.eval import eval_kernel_against_ref

class EvalTracesConfig(Config):
    def __init__(self):
        self.level = 1
        self.input_dir = f"reasoning_traces/level_{self.level}"
        self.dataset_dir = f"KernelBench/level{self.level}"
        self.num_gpus = torch.cuda.device_count()

def process_chunk(args: Tuple[List[str], EvalTracesConfig, int]):
    files, config, device_id = args
    print(f"Process started on GPU {device_id}")
    
    for file in files:
        print(f"Processing {file} on GPU {device_id}")
        file_path = os.path.join(config.input_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            problem_name = data["problem_name"]
            problem_id = data["problem_id"]
            ref_arch_src = read_file(os.path.join(config.dataset_dir, f"{problem_name}"))
            for i, generation in enumerate(data["generations"]):
                content = generation["content"]
                parse_pattern = r"^.*?</think>.*?```(.*?)```.*?$"
                match = re.match(parse_pattern, content, re.DOTALL)
                if match is None:
                    continue
                custom_cuda = match.group(1).strip()
                if "__global__" not in custom_cuda or "load_inline(" not in custom_cuda:
                    print(f"\n[load_inline not found in {file} for generation {i}]")
                    continue
                for code_type in ["python", "cpp"]:
                    if custom_cuda.startswith(code_type):
                        custom_cuda = custom_cuda[len(code_type):].strip()
                eval_result = eval_kernel_against_ref(
                    original_model_src=ref_arch_src,
                    custom_model_src=custom_cuda,
                    measure_performance=True,
                    device=device_id
                )
                
                if eval_result is not None:
                    print(f"Kernel for {file} generation {i} on GPU {device_id}: compiled={eval_result.compiled}, correctness={eval_result.correctness}")
                    generation["eval_result"] = {
                        "compiled": eval_result.compiled,
                        "correctness": eval_result.correctness,
                        "metadata": eval_result.metadata,
                        "runtime": eval_result.runtime,
                        "runtime_stats": eval_result.runtime_stats,
                        "runtime_original": eval_result.runtime_original,
                        "runtime_stats_original": eval_result.runtime_stats_original
                    }
                else:
                    generation["eval_result"] = None
        
        eval_dir = os.path.join(config.input_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, file), "w") as f:
            json.dump(data, f, indent=2)

@pydra.main(base=EvalTracesConfig)
def main(config: EvalTracesConfig):
    print(f"Evaluating traces for level {config.level}...")
    files = sorted(list(os.listdir(config.input_dir)))
    
    num_gpus = min(config.num_gpus, len(files))
    print(f"Using {num_gpus} GPUs for parallel processing")
    
    chunks = [[] for _ in range(num_gpus)]
    for i, file in enumerate(files):
        chunks[i % num_gpus].append(file)
    
    process_args = [(chunk, config, i) for i, chunk in enumerate(chunks)]
    
    with mp.Pool(num_gpus) as pool:
        pool.map(process_chunk, process_args)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()