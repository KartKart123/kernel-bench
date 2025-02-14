import torch
from src.eval import KernelExecResult, eval_kernel_against_ref
from src.utils import extract_first_code

def calculate_kernel_reward(
    eval_result: KernelExecResult,
    baseline_runtime: float,
    correctness_weight: float = 0.7,
    performance_weight: float = 0.3,
) -> float:
    correctness_reward = float(eval_result.correctness)
    
    if eval_result.correctness and eval_result.runtime > 0:
        speedup = baseline_runtime / eval_result.runtime
        performance_reward = speedup 
    else:
        performance_reward = 0.0
        
    total_reward = (correctness_weight * correctness_reward + 
                   performance_weight * performance_reward)
    
    return total_reward

def reward_fn(completions, ref_arch_src, baseline_runtime, correctness_weight=0.7, performance_weight=0.3, **kwargs):
    print(completions)
    print(ref_arch_src)
    input()
    rewards = []
    for completion, ref_arch in zip(completions, ref_arch_src):
        custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
        if custom_cuda is not None:
            eval_result = eval_kernel_against_ref(
                original_model_src=ref_arch,
                custom_model_src=custom_cuda,
                measure_performance=True
            )
            reward = calculate_kernel_reward(
                eval_result=eval_result,
                baseline_runtime=baseline_runtime,
                correctness_weight=correctness_weight,
                performance_weight=performance_weight
            )
            rewards.append(reward.total_reward)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards)