import torch
import re
from src.eval import KernelExecResult, eval_kernel_against_ref
from src.utils import extract_first_code

def calculate_kernel_reward(
    eval_result: KernelExecResult,
    baseline_runtime: float
) -> float:
    if eval_result is None:
        return 0.0
    correctness_reward = float(eval_result.correctness)
    if eval_result.correctness and eval_result.runtime > 0:
        speedup = baseline_runtime / eval_result.runtime
        performance_reward = speedup 
    else:
        performance_reward = 0.0
        
    total_reward = correctness_reward + performance_reward
    
    return total_reward

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_fn(prompts, completions, ref_arch_src, baseline_runtime, **kwargs):
    rewards = []
    pattern = r"^.*?</think>.*?```(.*?)```.*?$"
    #pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    for prompt, completion, runtime, ref_arch in zip(prompts, completions, baseline_runtime, ref_arch_src):
        reward = 0.0
        content = completion[0]["content"]
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        #print(content)
        #input()
        if match is None:
            rewards.append(reward)
            continue
        reward += 0.5

        #custom_cuda = extract_first_code(match.group(1), ["python", "cpp"])
        custom_cuda = match.group(1).strip()
        for code_type in ["python", "cpp"]:
            if custom_cuda.startswith(code_type):
                custom_cuda = custom_cuda[len(code_type) :].strip()
        #print(custom_cuda)
        #input()
        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch,
            custom_model_src=custom_cuda,
            measure_performance=True
        )
        print(eval_result)
        #input()
        reward += calculate_kernel_reward(
            eval_result=eval_result,
            baseline_runtime=runtime
        )
        rewards.append(reward)

    return rewards