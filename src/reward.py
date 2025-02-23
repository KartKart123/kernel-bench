import torch
import re
import os
import json
from src.safe_eval import KernelExecResult
import subprocess
import sys
import tempfile

def calculate_kernel_reward(
    eval_result: KernelExecResult
) -> float:
    if eval_result is None:
        return 0.0
    compilation_reward = float(eval_result.compiled)/2
    correctness_reward = float(eval_result.correctness)
    if eval_result.correctness and eval_result.runtime > 0:
        speedup = eval_result.runtime_original / eval_result.runtime
        performance_reward = speedup 
    else:
        performance_reward = 0.0
  
    # total_reward = compilation_reward +correctness_reward + performance_reward
    
    return (compilation_reward, correctness_reward, performance_reward)

def compute_format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*'''python\s*.*?'''\s*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_fn(prompts, completions, ref_arch_src, baseline_runtime, level, task_id, trainer, output_dir="outputs", **kwargs):
    process_index = trainer.accelerator.process_index
    # print(f"[Reward {process_index}] STARTING")
    os.makedirs(output_dir, exist_ok=True)
    rewards = []
    current_step = trainer.state.global_step
    device = trainer.model.device
    parse_pattern = r"^.*?</think>.*?```(.*?)```.*?$" #TODO change it
    format_pattern = r"^<think>.*?</think>\s*'''python\s*.*?'''\s*$"

    # Make cache directory for eval results
    eval_cache_dir = f"{output_dir}/eval_cache"
    eval_cache_path = f"{eval_cache_dir}/eval_results_{process_index}.json"
    os.makedirs(eval_cache_dir, exist_ok=True)

    for prompt, completion, ref_arch, ind_level, id in zip(prompts, completions, ref_arch_src, level, task_id):
        reward = 0.0
        content = completion[0]["content"]
        match = re.match(parse_pattern, content, re.DOTALL)
        # print(content)
        # input()
        if match is None:
            print(f"PROCESS {process_index} had no match")
            rewards.append(reward)
            continue

        format_match = re.match(format_pattern, content, re.DOTALL)
        format_reward = 1.0 if format_match else 0.0 # Just for saving to output; Won't be added to reward

        #custom_cuda = extract_first_code(match.group(1), ["python", "cpp"])
        custom_cuda = match.group(1).strip()
        for code_type in ["python", "cpp"]:
            if custom_cuda.startswith(code_type):
                custom_cuda = custom_cuda[len(code_type) :].strip()
    
        # print(f"[Reward {process_index}] main process output kernel (custom_cuda):")
        # print(custom_cuda)

        if ("__global__" not in custom_cuda) or ("load_inline(" not in custom_cuda) or ("try:" in content) or ("pass" in content):
            rewards.append(reward)
            print(f"[Reward {process_index}] output has no cuda kernel")
            continue

        reward += 0.5

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
                [sys.executable, "src/eval_script.py", ref_path, custom_path, str(process_index), eval_cache_path],
                timeout=120,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            process_out = result.stdout
            process_err = result.stderr

            # Parse CUDA compilation errors
            cuda_compilation_error_messages = []
            lines = process_out.split('\n')
            i = 0
            while i < len(lines):
                if "error: " in lines[i]:
                    error_start = lines[i].find("error: ") + len("error: ")
                    error_message = lines[i][error_start:]
                    
                    # Keep adding lines until we find one containing "/home"
                    j = i + 1
                    while j < len(lines) and "/home/ubuntu" not in lines[j]:
                        error_message += " " + lines[j]
                        j += 1
                    
                    cuda_compilation_error_messages.append(error_message.strip())
                    i = j
                else:
                    i += 1

            if result.returncode != 0:
                print(f"[Reward {process_index}] Evaluation failed with return code {result.returncode}")
            else:
                # Read results from a temporary file
                with open(eval_cache_path, 'r') as f:
                    output = json.load(f)
                os.unlink(eval_cache_path)  # Clean up the temp file
                output["metadata"]["cuda_compilation_error_messages"] = cuda_compilation_error_messages

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
        print(eval_result)

        compilation_reward, correctness_reward, performance_reward = calculate_kernel_reward( # Correctness and performance reward
            eval_result=eval_result
        )

        # Compute total reward
        reward += compilation_reward + correctness_reward + performance_reward
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
            "runtime": eval_result.runtime,
            "format_reward": format_reward,
            "compilation_reward": compilation_reward,
            "correctness_reward": correctness_reward,
            "performance_reward": performance_reward,
            "reward": reward + format_reward,
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
