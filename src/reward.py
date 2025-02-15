import torch
import re
from src.eval import KernelExecResult, eval_kernel_against_ref
from src.utils import extract_first_code

def calculate_kernel_reward(
    eval_result: KernelExecResult,
    baseline_runtime: float
) -> float:
    correctness_reward = float(eval_result.correctness)
    
    if eval_result.correctness and eval_result.runtime > 0:
        speedup = baseline_runtime / eval_result.runtime
        performance_reward = speedup 
    else:
        performance_reward = 0.0
        
    total_reward = correctness_reward + performance_reward
    
    return total_reward

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def reward_fn(prompts, completions, ref_arch_src, baseline_runtime, **kwargs):
    rewards = []
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    for prompt, completion, ref_arch in zip(prompts, completions, ref_arch_src):
        print(prompt)
        print("\n\n")
        print(completion)
        input()

        reward = 0.0
        content = completion[0]["content"]
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)

        if match is None:
            rewards.append(reward)
            continue
        reward += 0.5

        answer = extract_xml_answer(content)
        custom_cuda = extract_first_code(answer, ["python", "cpp"])

        if custom_cuda is None:
            rewards.append(reward)
            continue
        reward += 0.5

        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch,
            custom_model_src=custom_cuda,
            measure_performance=True
        )
        reward += calculate_kernel_reward(
            eval_result=eval_result,
            baseline_runtime=baseline_runtime
        )
        rewards.append(reward)

    return rewards