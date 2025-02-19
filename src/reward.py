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
    compilation_reward = float(eval_result.compiled)
    correctness_reward = float(eval_result.correctness)
    if eval_result.correctness and eval_result.runtime > 0:
        speedup = baseline_runtime / eval_result.runtime
        performance_reward = speedup 
    else:
        performance_reward = 0.0
        
    # total_reward = compilation_reward +correctness_reward + performance_reward
    
    return (compilation_reward, correctness_reward, performance_reward)

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_fn(trainer, prompts, completions, ref_arch_src, baseline_runtime, task_id, **kwargs):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True) # TODO: put into train_grpo.py
    rewards = []
    pattern = r"^.*?</think>.*?```(.*?)```.*?$"
    current_step = trainer.state.global_step
    device = trainer.model.device
    #pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    for prompt, completion, runtime, ref_arch, id in zip(prompts, completions, baseline_runtime, ref_arch_src, task_id):
        reward = 0.0
        content = completion[0]["content"]
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        # print(content)
        # input()
        if match is None:
            rewards.append(reward)
            continue
        parse_reward = 0.5

        #custom_cuda = extract_first_code(match.group(1), ["python", "cpp"])
        custom_cuda = match.group(1).strip()
        for code_type in ["python", "cpp"]:
            if custom_cuda.startswith(code_type):
                custom_cuda = custom_cuda[len(code_type) :].strip()
        # print(custom_cuda)
        # input()
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch,
            custom_model_src=custom_cuda,
            measure_performance=True,
            device=torch.device(device)
        )
        # print(eval_result)
        # input()
        compilation_reward, correctness_reward, performance_reward = calculate_kernel_reward( # Correctness and performance reward
            eval_result=eval_result,
            baseline_runtime=runtime
        )

        # Compute total reward
        reward = parse_reward + compilation_reward + correctness_reward + performance_reward
        rewards.append(reward)
        # print(reward)
        # input()

        # Save outputs
        arch_output_dir = f"{output_dir}/task_{id}"
        arch_output_path = f"{arch_output_dir}/step_{current_step}.json"
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
            "task_id": id,
            "prompt": prompt[0]["content"],
            "response": content,
            "compiled": eval_result.compiled,
            "correctness": eval_result.correctness,
            "runtime": eval_result.runtime,
            "baseline_runtime": runtime,
            "parse_reward": parse_reward,
            "compilation_reward": compilation_reward,
            "correctness_reward": correctness_reward,
            "performance_reward": performance_reward,
            "reward": reward,
        }
        data.append(entry)
        
        # Write back to file
        with open(arch_output_path, 'w') as f:
            json.dump(data, f,indent=2)

    return rewards