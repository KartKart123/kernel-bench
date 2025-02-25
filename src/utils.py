import re
import os
import torch
from functools import cache
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.eval import (
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    get_timing_stats,
    set_seed
)
from trl import ModelConfig


@cache
def load_deepseek_tokenizer():
    # TODO: Should we update this for new deepseek? Same tokenizer?
    # return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Instruct-0724")
    return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2", trust_remote_code=True)

# Buffer because deepseek totally blocks us if we send stuff that's too long :(
TOO_LONG_FOR_DEEPSEEK = 115_000


def is_safe_to_send_to_deepseek(prompt):
    tokenizer = load_deepseek_tokenizer()
    if type(prompt) == str:
        return (
            len(tokenizer(prompt, verbose=False)["input_ids"]) < TOO_LONG_FOR_DEEPSEEK
        )
    else:
        return len(tokenizer.apply_chat_template(prompt)) < TOO_LONG_FOR_DEEPSEEK

def set_gpu_arch(arch_list: list[str]):
    """
    Set env variable for torch cuda arch list to build kernels for specified architectures
    """
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(f"Invalid architecture: {arch}. Must be one of {valid_archs}")
    
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)

def measure_program_time(
        ref_arch_name: str,
        ref_arch_src: str, 
        num_trials: int = 100,
        use_torch_compile: bool = False,
        torch_compile_backend: str="inductor", 
        torch_compile_options: str="default",
        device: torch.device="cuda:0",
        verbose: bool = False,
):
    """
    Measure the time of a KernelBench reference architecture
    """
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Initialize PyTorch model, use this for eager mode execution
            model = Model(*init_inputs)
            
            if use_torch_compile:
                if verbose:
                    print(f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode")
                model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_options)
            elif verbose:
                print(f"Using PyTorch Eager Execution on {ref_arch_name}")
            
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_trials, verbose=verbose, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            print(f"{ref_arch_name} {runtime_stats}")
            
            return runtime_stats
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")


def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def print_messages(messages):
    for message in messages:
        print(message["role"])
        print(message["content"])
        print("-" * 50)
        print("\n\n")


def extract_python_code(text):
    """
    Extract python code from model output
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches) if matches else ""


def remove_code_block_header(code, code_language_type):
    """Assume input is code but just with like python, cpp, etc. at the top"""
    if code.startswith(code_language_type):
        code = code[len(code_language_type) :].strip()
    return code


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()
    else:
        code = trimmed

    # depends on code_language_type: cpp, python, etc.
    # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
    # in this case strip the cpp out
    for code_type in code_language_types:
       if code.startswith(code_type):
            code = code[len(code_type) :].strip()

    return code


def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None

def extract_code_blocks(text, code_language_types: list[str]) -> str:
    '''
    Extract all code blocks from text, combine them to return as a single string
    '''
    pattern = r'```.*?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type):].strip()
        combined_code.append(code)
    
    return " \n ".join(combined_code) if combined_code else ""

def get_tokenizer(
    model_name: str
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
    return tokenizer