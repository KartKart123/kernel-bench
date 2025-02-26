import torch
import re
import os
import json
from src.safe_eval import KernelExecResult
import subprocess
import sys
import tempfile
import shutil
import concurrent.futures
from typing import Tuple, Dict, Any, List

def calculate_kernel_reward(
    eval_result: KernelExecResult,
    baseline_runtime: float
) -> float:
    if eval_result is None:
        return (0.0, 0.0, 0.0, 0.0)
    COMPILED_COEFF = 0.5
    RUN_COEFF = 0.5
    CORRECTNESS_COEFF = 1
    SPEEDUP_COEFF = 1

    compilation_reward = float(eval_result.compiled) * COMPILED_COEFF
    run_reward = float(eval_result.run) * RUN_COEFF
    correctness_reward = float(eval_result.correctness) * CORRECTNESS_COEFF
    
    # if eval_result.correctness and eval_result.runtime > 0:
    #     speedup = baseline_runtime / eval_result.runtime
    #     performance_reward = speedup * SPEEDUP_COEFF
    # else:
    #     performance_reward = 0.0
  
    # total_reward = compilation_reward +correctness_reward + performance_reward
    performance_reward = 0.0
    
    return (compilation_reward, run_reward, correctness_reward, performance_reward)

# def compute_format_reward(completions, **kwargs):
#     pattern = r"^<think>.*?</think>\s*```python\s*.*?```\s*$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
#     return [0.5 if match else 0.0 for match in matches]

def _process_single_item(
    item_data: Dict[str, Any],
    parse_pattern: str,
    verbose: bool
) -> Tuple[float, Dict[str, Any]]:
    """Process a single item from the main loop and return its reward and output data."""
    prompt = item_data["prompt"]
    completion = item_data["completion"]
    ref_arch = item_data["ref_arch"]
    runtime = item_data["runtime"]
    ind_level = item_data["ind_level"]
    id = item_data["id"]
    process_index = item_data["process_index"]
    current_step = item_data["current_step"]
    device = item_data["device"]
    output_dir = item_data["output_dir"]
    eval_cache_path = item_data["eval_cache_path"]
    
    reward = 0.0
    content = completion[0]["content"]
    if verbose: 
        print("=" * 80)
        print(f"[Reward {process_index}] Task ID: {id}")
        print(content)

    match = re.match(parse_pattern, content, re.DOTALL)
    if match is None:
        print(f"[Reward {process_index} EXIT] had no match")
        print(content)
        return reward, None

    custom_cuda = match.group(1).strip() 
    for code_type in ["python", "cpp", "cuda"]:
        if custom_cuda.startswith(code_type):
            custom_cuda = custom_cuda[len(code_type) :].strip()

    if verbose:
        print(f"[Reward {process_index}] main process output kernel (custom_cuda):")
        print(custom_cuda)

    if ("__global__" not in custom_cuda) or ("load_inline(" not in custom_cuda) or ("try:" in custom_cuda) or ("pass" in custom_cuda):
        print(f"[Reward {process_index} EXIT] output has no cuda kernel")
        print(custom_cuda)
        return reward, None

    # Write code to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f_ref:
        f_ref.write(ref_arch)
        ref_path = f_ref.name
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f_custom:
        f_custom.write(custom_cuda)
        custom_path = f_custom.name

    eval_result = KernelExecResult()
    try:
        result = subprocess.run(
            [sys.executable, "src/safe_eval_script.py", ref_path, custom_path, str(process_index), eval_cache_path],
            timeout=120,
            stdout=None,
            stderr=None,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[Reward {process_index}] Evaluation failed with return code {result.returncode}")
        else:
            # Read results from a temporary file
            with open(eval_cache_path, 'r') as f:
                output = json.load(f)
            
            # Don't delete the cache file here - we'll do it after all workers finish
            
            eval_result = KernelExecResult(
                compiled=output["compiled"],
                run=output["run"],
                correctness=output["correctness"],
                metadata=output["metadata"],
                # # runtime=output["runtime"],
                # runtime_stats=output["runtime_stats"],
            )
            
    except subprocess.TimeoutExpired:
        print(f"[Reward {process_index}] Evaluation timed out")
    except Exception as e:
        print(f"[Reward {process_index}] Evaluation failed with error: {str(e)}")
    finally:
        # Clean up temporary files
        try:
            os.unlink(ref_path)
            os.unlink(custom_path)
        except:
            pass

    print(f"[Task {id}]: {eval_result}")

    compilation_reward, run_reward, correctness_reward, performance_reward = calculate_kernel_reward(
        eval_result=eval_result,
        baseline_runtime=runtime
    )

    # Compute total reward
    reward += compilation_reward + run_reward + correctness_reward + performance_reward

    # Prepare entry for output
    entry = {
        "level": ind_level,
        "task_id": id,
        "step": current_step,
        "device": device.index,
        "compiled": eval_result.compiled,
        "run": eval_result.run,
        "correctness": eval_result.correctness,
        "error_type": eval_result.metadata["error_type"] if "error_type" in eval_result.metadata else None,
        "error_msg": eval_result.metadata["error_msg"] if "error_msg" in eval_result.metadata else None,
        "runtime": eval_result.runtime,
        "baseline_runtime": runtime,
        "compilation_reward": compilation_reward,
        "run_reward": run_reward,
        "correctness_reward": correctness_reward,
        "performance_reward": performance_reward,
        "reward": reward,
        "prompt": prompt[0]["content"],
        "response": content,
    }
    
    return reward, entry

def reward_fn(prompts, completions, ref_arch_src, baseline_runtime, level, task_id, trainer, output_dir="outputs", **kwargs):
    base_process_index = trainer.accelerator.process_index
    current_step = trainer.state.global_step
    device = trainer.model.device
    parse_pattern = r"^.*?</think>.*?```(.*?)```.*?$" #TODO change it
    verbose = False

    # Make cache directory for eval results
    eval_cache_dir = f"{output_dir}/eval_cache"
    os.makedirs(eval_cache_dir, exist_ok=True)

    # Prepare batch of items for parallel processing
    items = []
    for idx, (prompt, completion, ref_arch, runtime, ind_level, id) in enumerate(zip(prompts, completions, ref_arch_src, baseline_runtime, level, task_id)):
        # Create a unique process index for each parallel task
        unique_process_index = f"{base_process_index}_{idx}"
        eval_cache_path = f"{eval_cache_dir}/eval_results_{unique_process_index}.json"
        
        item_data = {
            "prompt": prompt,
            "completion": completion,
            "ref_arch": ref_arch,
            "runtime": runtime,
            "ind_level": ind_level,
            "id": id,
            "process_index": unique_process_index,  # Use the unique process index
            "current_step": current_step,
            "device": device,
            "output_dir": output_dir,
            "eval_cache_path": eval_cache_path
        }
        items.append(item_data)

    # Process items in parallel
    max_workers = min(len(items), os.cpu_count() or 4)  # Use at most CPU count workers
    rewards = [0.0] * len(items)  # Pre-allocate rewards list with default values
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and store futures with their original indices
        futures = []
        for i, item in enumerate(items):
            future = executor.submit(_process_single_item, item, parse_pattern, verbose)
            futures.append((future, i))
            
        # Process results as they complete, but store in original order
        for future, original_idx in futures:
            try:
                reward, entry = future.result()
                rewards[original_idx] = reward  # Store at original index to preserve order
                
                # Save the result if we have valid data
                if entry is not None:
                    # Save outputs
                    ind_level = items[original_idx]["ind_level"]
                    arch_output_dir = f"{output_dir}/level_{ind_level}/step_{current_step}"
                    arch_output_path = f"{arch_output_dir}/device_{device.index}.json"
                    os.makedirs(arch_output_dir, exist_ok=True)
                    
                    # Initialize or load existing data
                    if os.path.exists(arch_output_path):
                        with open(arch_output_path, 'r') as f:
                            try:
                                data = json.loads(f.read())
                            except json.JSONDecodeError:
                                data = []
                    else:
                        data = []
                    
                    # Append new entry
                    data.append(entry)
                    
                    # Write back to file
                    with open(arch_output_path, 'w') as f:
                        json.dump(data, f, indent=2)
            except Exception as exc:
                print(f"Processing item {items[original_idx]['id']} generated an exception: {exc}")
                # rewards[original_idx] already has the default 0.0 value
                
    # Clean up temp files
    for item in items:
        try:
            if os.path.exists(item["eval_cache_path"]):
                os.unlink(item["eval_cache_path"])
        except:
            pass

    return rewards