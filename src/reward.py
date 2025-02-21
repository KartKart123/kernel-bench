import torch
import re
import os
import json
from src.eval import KernelExecResult, eval_kernel_against_ref

def calculate_kernel_reward(
    eval_result: KernelExecResult,
    baseline_runtime: float
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
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_fn(prompts, completions, ref_arch_src, baseline_runtime, level, task_id, trainer, output_dir="outputs", **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    rewards = []
    current_step = trainer.state.global_step
    device = trainer.model.device
    parse_pattern = r"^.*?</think>.*?```(.*?)```.*?$" #TODO change it
    format_pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    for prompt, completion, runtime, ref_arch, ind_level, id in zip(prompts, completions, baseline_runtime, ref_arch_src, level, task_id):
        reward = 0.0
        content = completion[0]["content"]
        match = re.match(parse_pattern, content, re.DOTALL | re.MULTILINE)
        # print(content)
        # input()
        if match is None:
            rewards.append(reward)
            continue

        format_match = re.match(format_pattern, content, re.DOTALL | re.MULTILINE)
        format_reward = 1.0 if format_match else 0.0 # Just for saving to output; Won't be added to reward

        #custom_cuda = extract_first_code(match.group(1), ["python", "cpp"])
        custom_cuda = match.group(1).strip()
        for code_type in ["python", "cpp"]:
            if custom_cuda.startswith(code_type):
                custom_cuda = custom_cuda[len(code_type) :].strip()

        if "__global__" not in custom_cuda or "load_inline(" not in custom_cuda:
            rewards.append(reward)
            continue

        #reward += 0.5

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

        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch,
            custom_model_src=custom_cuda,
            measure_performance=True,
            device=torch.device(device)
        )
        if trainer.accelerator.is_main_process:
            print(task_id)
        print(eval_result)
        # input()
        compilation_reward, correctness_reward, performance_reward = calculate_kernel_reward( # Correctness and performance reward
            eval_result=eval_result,
            baseline_runtime=runtime
        )

        # Compute total reward
        reward += compilation_reward + correctness_reward + performance_reward
        #reward += correctness_reward + performance_reward
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
            "baseline_runtime": runtime,
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

    return rewards
