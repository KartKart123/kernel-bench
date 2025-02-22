from prompt_constructor import custom_prompt_generate_custom_cuda
from enum import Enum, auto

class AnswerResult(Enum):
    FORMAT = auto()
    NO_CUDA = auto()
    COMPILATION = auto()
    RUNTIME = auto()
    CORRECTNESS = auto()
    SPEEDUP = auto()

def construct_formatting_answer(previous_answer, additional_info):
    prompt = f"""You have already attempted to answer, but failed due to not adhering to the desired formatting.
Your solution may be correct, but needs to be properly formatted with <think> and <answer> tags."""
    prompt += f'\nYour previous answer was:\n{previous_answer}'
    prompt += "\nRefine the existing code to achieve the desired formatting"
    return prompt

def construct_no_cuda_answer(previous_answer, additional_info):
    prompt = f"""You have already attempted to answer, but failed to include any CUDA kernel implementations.
Your solution uses only PyTorch operations without any custom CUDA kernels."""
    prompt += f'\nYour previous answer was:\n{previous_answer}'
    prompt += "\nRefine the existing code to include and use custom CUDA kernels"
    return prompt

def construct_compilation_answer(previous_answer, additional_info):
    prompt = f"""You have already attempted to answer, but your implementation failed to compile.
The code structure appears correct but there are syntax or semantic errors preventing compilation."""
    prompt += f'\nYour previous answer was:\n{previous_answer}'
    prompt += f'\nCompilation error:\n{additional_info}'
    prompt += "\nFix the compilation errors while maintaining the same optimization approach"
    return prompt

def construct_runtime_answer(previous_answer, additional_info):
    prompt = f"""You have already attempted to answer, and your implementation compiled successfully but had runtime errors.
The code syntax is valid but there are issues when executing the kernels."""
    prompt += f'\nYour previous answer was:\n{previous_answer}'
    prompt += f'\nRuntime error:\n{additional_info}'
    prompt += "\nFix the runtime errors while maintaining the same optimization strategy"
    return prompt

def construct_correctness_answer(previous_answer, additional_info):
    prompt = f"""You have already attempted to answer. Your implementation compiled and ran successfully, but produced incorrect results.
The kernels execute without errors but the output does not match the reference implementation."""
    prompt += f'\nYour previous answer was:\n{previous_answer}'
    prompt += "\nFix the correctness issues while maintaining the same optimization approach where possible"
    return prompt

def construct_speedup_answer(previous_answer, additional_info):
    prompt = f"""You have already attempted to answer. Your implementation compiled, ran successfully, and produced correct results, but did not achieve sufficient speedup compared to the baseline implementation."""
    prompt += f'\nYour previous answer was:\n{previous_answer}'
    prompt += f'Speedup was {additional_info}'
    prompt += "\nOptimize your solution further while maintaining correctness"
    return prompt

def custom_prompt_generate_refinement_prompt(arc_src, previous_answer, answer_type: AnswerResult, additional_info):
    base_prompt = custom_prompt_generate_custom_cuda(arc_src)
    
    match answer_type:
        case AnswerResult.FORMAT:
            refinement_additional_prompt = construct_formatting_answer(previous_answer, additional_info)
        case AnswerResult.NO_CUDA:
            refinement_additional_prompt = construct_no_cuda_answer(previous_answer, additional_info)
        case AnswerResult.COMPILATION:
            refinement_additional_prompt = construct_compilation_answer(previous_answer, additional_info)
        case AnswerResult.RUNTIME:
            refinement_additional_prompt = construct_runtime_answer(previous_answer, additional_info)
        case AnswerResult.CORRECTNESS:
            refinement_additional_prompt = construct_correctness_answer(previous_answer, additional_info)
        case AnswerResult.SPEEDUP:
            refinement_additional_prompt = construct_speedup_answer(previous_answer, additional_info)
        case _:
            raise ValueError(f"Unknown answer type: {answer_type}")

    return base_prompt + '\n' + refinement_additional_prompt

def add_refinement_prompt_to_dataset(dataset, ref_arc_src, previous_answer, answer_type, additional_info):
    train_prompt = custom_prompt_generate_refinement_prompt(ref_arc_src, previous_answer, answer_type, additional_info)
    dataset.add_datapoint(train_prompt, ref_arc_src, baseline_runtime, level, task_id)
