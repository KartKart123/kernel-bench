from pydra import Config
import os
import re
import pydra
import json
import random
from src.prompt_constructor import custom_prompt_generate_custom_cuda
from src.utils import read_file
import torch
import multiprocessing as mp
from typing import List, Dict, Tuple
from src.eval import eval_kernel_against_ref, KernelExecResult
import subprocess
import sys
import tempfile

class EvalTracesConfig(Config):
    def __init__(self):
        self.level = 1
        self.input_dir = f"reasoning_traces/level_{self.level}"
        self.dataset_dir = f"KernelBench/level{self.level}"
        self.num_gpus = torch.cuda.device_count()

def process_chunk(args: Tuple[List[str], EvalTracesConfig, int]):
    files, config, device_id = args
    print(f"Process started on GPU {device_id}")
     # Make cache directory for eval results
    eval_cache_dir = f"eval_trace_cache"
    eval_cache_path = f"{eval_cache_dir}/eval_results_{device_id}.json"
    os.makedirs(eval_cache_dir, exist_ok=True)
    
    for file in files:
        print(f"Processing {file} on GPU {device_id}")
        file_path = os.path.join(config.input_dir, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            problem_name = data["problem_name"]
            problem_id = data["problem_id"]
            ref_path = os.path.join(config.dataset_dir, f"{problem_name}")
            ref_arch_src = read_file(ref_path)
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
                # eval_result = eval_kernel_against_ref(
                #     original_model_src=ref_arch_src,
                #     custom_model_src=custom_cuda,
                #     measure_performance=True,
                #     device=device_id
                # )
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f_custom:
                    f_custom.write(custom_cuda)
                    custom_path = f_custom.name
                eval_result = None
                try:
                    result = subprocess.run(
                        [sys.executable, "scripts/eval_traces_script.py", ref_path, custom_path, str(device_id), eval_cache_path],
                        timeout=120,
                        stdout=None,  # Let all output pass through
                        stderr=None,
                        text=True
                    )

                    if result.returncode != 0:
                        print(f"[Device {device_id}] Evaluation failed with return code {result.returncode}")
                    else:
                        # Read results from a temporary file
                        with open(eval_cache_path, 'r') as f:
                            output = json.load(f)
                        os.unlink(eval_cache_path)  # Clean up the temp file

                        eval_result = KernelExecResult(
                            compiled=output["compiled"],
                            correctness=output["correctness"],
                            metadata=output["metadata"],
                            runtime=output["runtime"],
                            runtime_stats=output["runtime_stats"],
                            runtime_original=output["runtime_original"],
                            runtime_stats_original=output["runtime_stats_original"]
                        )
                        
                except subprocess.TimeoutExpired:
                    print(f"[Device {device_id}] Evaluation timed out")
                except Exception as e:
                    print(f"[Device {device_id}] Evaluation failed with error: {str(e)}")
                finally:
                    # Clean up temporary files
                    try:
                        # os.unlink(ref_path)
                        os.unlink(custom_path)
                    except:
                        pass

                if eval_result is not None:
                    print(f"Kernel for {file} generation {i} on GPU {device_id}: compiled={eval_result.compiled}, correctness={eval_result.correctness}")
                    generation["eval_result"] = {
                        "compiled": eval_result.compiled,
                        "correctness": eval_result.correctness,
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
    cnt = 0
    # finished_list = [1, 2, 4, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 22, 24, 26, 27, 28, 29, 31, 34, 35, 36, 41, 42, 43, 49, 50, 56, 57, 63, 64, 70, 78, 100]
    todo_list = [6, 7, 8, 9, 12, 16, 17, 18, 48, 79, 86, 87, 89, 90, 92, 93, 94, 96, 97, 98]
    for i, file in enumerate(files):
        if file.startswith('eval'):
            continue
        idx = int(file.split('_')[0])
        if idx in todo_list:
            chunks[cnt % num_gpus].append(file)
            cnt += 1
    
    process_args = [(chunk, config, i) for i, chunk in enumerate(chunks)]
    
    with mp.Pool(num_gpus) as pool:
        pool.map(process_chunk, process_args)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()