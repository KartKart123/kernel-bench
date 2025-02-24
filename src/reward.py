import torch
import re
import os
import json
from src.safe_eval import KernelExecResult
import subprocess
import sys
import tempfile

def calculate_kernel_reward(
    eval_result: KernelExecResult,
    # baseline_runtime: float
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
    
    if eval_result.correctness and eval_result.runtime > 0:
        speedup = eval_result.runtime_original / eval_result.runtime
        performance_reward = speedup * SPEEDUP_COEFF
    else:
        performance_reward = 0.0
  
    # total_reward = compilation_reward +correctness_reward + performance_reward
    
    return (compilation_reward, run_reward, correctness_reward, performance_reward)

# def compute_format_reward(completions, **kwargs):
#     pattern = r"^<think>.*?</think>\s*```python\s*.*?```\s*$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
#     return [0.5 if match else 0.0 for match in matches]

def reward_fn(prompts, completions, ref_arch_src, level, task_id, trainer, output_dir="outputs", **kwargs):
    process_index = trainer.accelerator.process_index
    rewards = []
    current_step = trainer.state.global_step
    device = trainer.model.device
    parse_pattern = r"^.*?</think>.*?```(.*?)```.*?$" #TODO change it
    # format_pattern = r"^<think>.*?</think>\s*```python\s*.*?```\s*$"
    verbose = False

    # Make cache directory for eval results
    eval_cache_dir = f"{output_dir}/eval_cache"
    eval_cache_path = f"{eval_cache_dir}/eval_results_{process_index}.json"
    os.makedirs(eval_cache_dir, exist_ok=True)

    for prompt, completion, ref_arch, ind_level, id in zip(prompts, completions, ref_arch_src, level, task_id):
        # print(f"[Reward {process_index}] Processing task {id}")
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
            rewards.append(reward)
            continue

        # format_match = re.match(format_pattern, content, re.DOTALL)
        # if format_match is None:
        #     print(f"[Reward {process_index}] had no format match")
        # format_reward = 0.5 if format_match else 0.0 # Just for saving to output; Won't be added to reward

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
            rewards.append(reward)
            continue

        # reward += 0.5 

        # Save response before evaluation
        # pre_eval_dir = f"{output_dir}/level_{ind_level}/step_{current_step}/pre_eval"
        # os.makedirs(pre_eval_dir, exist_ok=True)
        # pre_eval_path = f"{pre_eval_dir}/device_{device.index}_{id}.json"
        
        # pre_eval_data = {
        #     "level": ind_level,
        #     "task_id": id,
        #     "step": current_step,
        #     "device": device.index,
        #     "prompt": prompt[0]["content"],
        #     "response": content,
        #     "custom_cuda": custom_cuda
        # }
        
        # with open(pre_eval_path, 'w') as f:
        #     json.dump(pre_eval_data, f, indent=2)

        # print(f"[Reward {process_index}] EVALUATING")
        
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
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                stdout=None,
                stderr=None,
                text=True
            )
            
            # process_out = result.stdout
            # process_err = result.stderr

            # # Parse CUDA compilation errors using regex
            # error_pattern = r"error: (.*?)(?=/home)"
            # error_messages = re.findall(error_pattern, process_out, re.DOTALL)
            # error_messages = [msg.strip() for msg in error_messages]

            if result.returncode != 0:
                print(f"[Reward {process_index}] Evaluation failed with return code {result.returncode}")
            else:
                # Read results from a temporary file
                with open(eval_cache_path, 'r') as f:
                    output = json.load(f)
                os.unlink(eval_cache_path)  # Clean up the temp file
                # output["metadata"]["cuda_compilation_error_messages"] = error_messages

                eval_result = KernelExecResult(
                    compiled=output["compiled"],
                    run=output["run"],
                    correctness=output["correctness"],
                    metadata=output["metadata"],
                    runtime=output["runtime"],
                    runtime_stats=output["runtime_stats"],
                    runtime_original=output["runtime_original"],
                    runtime_stats_original=output["runtime_stats_original"]
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

        # print(f"[Reward {process_index}] FINISHED EVAL")
        # print(f"[Reward {process_index}] EVAL RESULT: {eval_result}")
        print(f"[Task {id}]: {eval_result}")

        compilation_reward, run_reward, correctness_reward, performance_reward = calculate_kernel_reward( # Correctness and performance reward
            eval_result=eval_result
        )

        # Compute total reward
        reward += compilation_reward + run_reward + correctness_reward + performance_reward
        rewards.append(reward)
        # print(reward)
        # input()

        # Save outputs
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
        entry = {
            "level": ind_level,
            "task_id": id,
            "step": current_step,
            "device": device.index,
            "compiled": eval_result.compiled,
            "correctness": eval_result.correctness,
            "error_type": eval_result.metadata["error_type"] if "error_type" in eval_result.metadata else None,
            "error_msg": eval_result.metadata["error_msg"] if "error_msg" in eval_result.metadata else None,
            "runtime": eval_result.runtime,
            # "format_reward": format_reward,
            "compilation_reward": compilation_reward,
            "run_reward": run_reward,
            "correctness_reward": correctness_reward,
            "performance_reward": performance_reward,
            "reward": reward,
            "prompt": prompt[0]["content"],
            "response": content,
        }
        data.append(entry)

        # Write back to file
        with open(arch_output_path, 'w') as f:
            json.dump(data, f,indent=2)
        # print(f"[Reward {process_index}] SAVING OUTPUTS DONE")
    # print(f"[Reward {process_index}] REWARDS: {rewards}")
    return rewards
