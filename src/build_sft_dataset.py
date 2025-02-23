from pydra import Config
import os
import re
import json
import pydra
import numpy as np
from collections import defaultdict
from src.prompt_constructor import custom_prompt_generate_custom_cuda
from src.utils import read_file

class BuildConfig(Config):
    def __init__(self):
        self.level = 1
        self.eval_dir = f"reasoning_traces/level_{self.level}/eval"
        self.dataset_dir = f"KernelBench/level{self.level}"
        self.output_dir = "data/sft"
        self.min_speedup = 0.0 
        self.output_file = f"kernelbench_sft_level_{self.level}.json"
        self.top_k = 3 
        self.top_k_output_file = f"kernelbench_sft_level_{self.level}_top{self.top_k}.json"

def process_file(file_path: str, config: BuildConfig):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    problem_name = data["problem_name"]
    ref_arch_src = read_file(os.path.join(config.dataset_dir, problem_name))
    
    successful_examples = []
    for gen in data['generations']:
        if 'eval_result' not in gen or gen['eval_result'] is None:
            continue
            
        result = gen['eval_result']
        if not (result['compiled'] and result.get('correctness', False)):
            continue
            
        if not (result.get('runtime') and result.get('runtime_original')):
            continue
            
        speedup = result['runtime_original'] / result['runtime']
        if speedup < config.min_speedup:
            continue
            
        content = gen['content']
        parse_pattern = r"^.*?</think>.*?```(.*?)```.*?$"

        match = re.match(parse_pattern, content, re.DOTALL)
        if not match:
            continue
            
        cuda_code = match.group(1).strip()
        for code_type in ["python", "cpp"]:
            if cuda_code.startswith(code_type):
                cuda_code = cuda_code[len(code_type):].strip()
        
        prompt = custom_prompt_generate_custom_cuda(ref_arch_src)
        example = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"```python\n{cuda_code}\n```"}
            ],
            "problem_name": problem_name,
            "runtime": result['runtime'],
            "speedup": speedup
        }
        successful_examples.append(example)
    
    return successful_examples

def get_top_k_examples(examples, k):
    problem_examples = defaultdict(list)
    for ex in examples:
        problem_examples[ex['problem_name']].append(ex)
    
    top_k_examples = []
    for problem, prob_examples in problem_examples.items():
        sorted_examples = sorted(prob_examples, key=lambda x: x['runtime'])
        top_k_examples.extend(sorted_examples[:k])
    
    return top_k_examples

@pydra.main(base=BuildConfig)
def main(config: BuildConfig):
    print(f"Building SFT dataset from evaluation results in {config.eval_dir}...")
        
    files = sorted(os.listdir(config.eval_dir))
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    all_examples = []
    
    for file in files:
        file_path = os.path.join(config.eval_dir, file)
        examples = process_file(file_path, config)
        all_examples.extend(examples)
    
    output_path = os.path.join(config.output_dir, config.output_file)
    with open(output_path, 'w') as f:
        json.dump(all_examples, f, indent=2)
    
    top_k_examples = get_top_k_examples(all_examples, config.top_k)
    top_k_output_path = os.path.join(config.output_dir, config.top_k_output_file)
    with open(top_k_output_path, 'w') as f:
        json.dump(top_k_examples, f, indent=2)
    
    print(f"Total examples collected: {len(all_examples)}")
    
    print(f"Total examples in top-{config.top_k} dataset: {len(top_k_examples)}")

if __name__ == "__main__":
    main()
