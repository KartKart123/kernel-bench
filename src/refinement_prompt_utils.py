from src.prompt_constructor import custom_prompt_generate_custom_cuda
from enum import Enum, auto
from datasets import Dataset
import os
import json
import shutil
from datasets import load_dataset, Dataset, IterableDataset, concatenate_datasets



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
Your solution somehow avoids using the CUDA kernel, either by not including it or by not actually using it."""
    prompt += f'\nThis is your previous answer:\n{previous_answer}'
    prompt += "\nRefine the existing code to include and use custom CUDA kernels"
    prompt += "!!!!"
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

def get_refined_prompt(arc_src, previous_answer, answer_type, additional_info):
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

def row_in_dataset(dataset, row):
    filtered = dataset.filter(
        lambda example: all(
            example[key] == value 
            for key, value in row.items()
        )
    )
    return len(filtered) > 0

# def init_dataset(info_dict, path, cleanup=False) -> Dataset:
#     assert path == "dataset/", "Just making sure you are not deleting what you're not supposed to : ) Remove this assert if you're sure the path is right"
#     dataset = Dataset.from_dict(info_dict)
#     dataset.path = path
#     if cleanup and os.path.exists(path):
#         shutil.rmtree(path, ignore_errors=True)
#     os.makedirs(path, exist_ok=True)
#     for prompt, ref_arch_src, level, task_id in zip(info_dict["prompt"], info_dict["ref_arch_src"], info_dict["level"], info_dict["task_id"]):
#         dataset.add_datapoint(info_dict)
#     return dataset
import random
def sync_dataset(dataset: Dataset) -> Dataset:
    """
    Synchronize the dataset by reading all files from disk and creating a fresh dataset.
    Forces a complete refresh of the dataset to ensure all processes see the latest data.
    """
    # Initialize empty lists for each field
    new_data = {
        "prompt": [],
        "ref_arch_src": [],
        "level": [],
        "task_id": []
    }

    for filename in os.listdir(dataset.path):
        if filename.endswith('.json'):
            filepath = os.path.join(dataset.path, filename)
            with open(filepath, 'r') as f:
                try:
                    content = json.load(f)
                    new_data["prompt"].append(content.get("prompt"))
                    new_data["ref_arch_src"].append(content.get("ref_arch_src"))
                    new_data["level"].append(content.get("level"))
                    new_data["task_id"].append(content.get("task_id"))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
                    continue
        else:
            print(f"Skipping non-JSON file: {filename}")

    new_dataset = Dataset.from_dict(new_data)
    new_dataset.path = dataset.path
    new_dataset.set_format(type=None)
    
    print(f"Synced dataset contains {len(new_dataset)} items")
    
    return new_dataset
import os
import json
import hashlib
from datasets import Dataset

def compute_hash(content: dict) -> str:
    """Compute SHA-256 hash of the JSON content."""
    content_bytes = json.dumps(content, sort_keys=True).encode('utf-8')
    return hashlib.sha256(content_bytes).hexdigest()

# def add_datapoint(dataset: Dataset, prompt: str, ref_arch_src: str, level: int, task_id: int) -> None:

#     prompt = prompt if isinstance(prompt, list) else [{"role" : "user", "content" : prompt}]
#     new_data = {
#         "prompt": prompt,
#         "ref_arch_src": ref_arch_src,
#         "level": level,
#         "task_id": task_id
#     }

#     data_hash = compute_hash(new_data)

#     existing_files = [f for f in os.listdir(dataset.path) if f.startswith(f"{level}_{task_id}_") and f.endswith('.json')]
#     existing_hashes = set()
#     max_refinement_level = -1

#     for filename in existing_files:
#         filepath = os.path.join(dataset.path, filename)
#         with open(filepath, 'r') as f:
#             try:
#                 content = json.load(f)
#             except json.JSONDecodeError:
#                 print(filepath)
#                 raise JSONDecodeError
#             file_hash = compute_hash(content)
#             existing_hashes.add(file_hash)
#             refinement_level = int(filename.split('_')[2].split('.')[0])
#             if refinement_level > max_refinement_level:
#                 max_refinement_level = refinement_level

#     if data_hash in existing_hashes:
#         print("Duplicate data found. Skipping addition.")
#         return

#     new_refinement_level = max_refinement_level + 1

#     # Create the new filename with hash
#     filename = f"{level}_{task_id}_{new_refinement_level}_{data_hash[:8]}.json"
#     filepath = os.path.join(dataset.path, filename)

#     os.makedirs(os.path.dirname(filepath), exist_ok=True)
#     with open(filepath, 'w') as f:
#         json.dump(new_data, f)

# def base_criteria_fn(content):
#     '''By default, do not remove anything'''
#     return False

# def clean_dataset(dataset: Dataset, should_be_removed = base_criteria_fn) -> Dataset:
#     '''!!This shit is NOT in-place!!'''
#     path = dataset.path
#     files = os.listdir(path)
#     to_add = []
#     for filename in files:
#         if not filename.endswith('.json'):
#             raise ValueError("you're not supposed to have non-json files")
#         filepath = os.path.join(path, filename)
#         with open(filepath, 'r') as f:
#             content = json.load(f)
            
#         if not should_be_removed(content):
#             os.remove(filepath)
#         else:
#             to_add.append(content)
#     dataset = Dataset.from_dict({}).add_items(to_add)
#     dataset.path = dataset
#     return dataset


import os
import json
from datasets import Dataset
from typing import Dict, Any

import os
import json
import shutil
from datasets import Dataset
from typing import Dict, Any, List

import os
import json
import shutil
from datasets import Dataset
from typing import Dict, Any, List

class MutableDataset(Dataset):
    def __init__(self, starting_data = None, path: str = "dataset/", cleanup: bool = False):
        if cleanup and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
        self._data = {}
        self.path = path
        if starting_data is not None:
            for a,b,c,d in zip(starting_data["prompt"], starting_data["ref_arch_src"], starting_data["level"], starting_data["task_id"]):
                self.add_datapoint(a,b,c,d)
        self.sync()

    def __len__(self):
        return len(next(iter(self._data.values()), []))

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        
        result = {key: [] for key in self._data.keys()}
        for _ in range(len(indices)):
            idx = random.randint(0, len(self._data["prompt"]) - 1)
            for key in self._data.keys():
                result[key].append(self._data[key][idx])
        
        return result

    def add_datapoint(self, prompt: str, ref_arch_src: str, level: int, task_id: int):
        p = prompt if isinstance(prompt, list) else [{"role" : "user", "content" : prompt}]
        new_data = {
            "prompt": p,
            "ref_arch_src": ref_arch_src,
            "level": level,
            "task_id": task_id
        }
        filename = os.path.join(self.path, f"{random.randint(1,100000000)}.json")
        with open(filename, 'w') as f:
            json.dump(new_data, f)

    def sync(self):
        aggregated_data = {}
        for filename in os.listdir(self.path):
            if filename.endswith('.json'):
                with open(os.path.join(self.path, filename), 'r') as f:
                    content = json.load(f)
                    for key, value in content.items():
                        aggregated_data.setdefault(key, []).append(value)
        self._data = aggregated_data


def test_dynamic_dataset():
    # Test data
    train_prompts = ["prompt_1", "prompt_2", "prompt_3"]
    ref_arch_srcs = ["source_1", "source_2", "source_3"]
    levels = [1, 2, 3]
    task_ids = [0, 1, 2]

    print("Making Dataset")
    dataset = Dataset.from_dict({
        "prompt" : train_prompts,
        "ref_arch_src" : ref_arch_srcs,
        "level" : levels,
        "task_id" : task_ids,
    })
    dataset.path = "testing"
    os.makedirs("testing", exist_ok=True)

    # Test initial dataset size
    print(f"Initial dataset size: {len(dataset)}")
    assert len(dataset) == 3, "Initial dataset size should be 3"

    # Test accessing elements (random due to shuffle=True)
    for _ in range(5):
        item = dataset[0]
        print(f"Random item: {item}")
        assert item["prompt"] in train_prompts, "Prompt should be in the initial prompts"
        assert item["ref_arch_src"] in ref_arch_srcs, "Source should be in the initial sources"
        assert item["level"] in levels, "Level should be in the initial levels"
        assert item["task_id"] in task_ids, "Task ID should be in the initial task IDs"

    # Test adding a new datapoint
    add_datapoint(dataset, "prompt_4", "source_4", 4, 3)
    print(f"Dataset size after adding a new datapoint: {len(dataset)}")
    assert len(dataset) == 4, "Dataset size should be 4 after adding a new datapoint"

    # Test accessing the new datapoint
    new_item = dataset[3]  # Access the last item (newly added)
    print(f"New item: {new_item}")
    assert new_item["prompt"] == "prompt_4", "New prompt should be 'prompt_4'"
    assert new_item["ref_arch_src"] == "source_4", "New source should be 'source_4'"
    assert new_item["level"] == 4, "New level should be 4"
    assert new_item["task_id"] == 3, "New task ID should be 3"

    # Test adding another datapoint
    add_datapoint(dataset, "prompt_5", "source_5", 5, 4)
    print(f"Dataset size after adding another datapoint: {len(dataset)}")
    assert len(dataset) == 5, "Dataset size should be 5 after adding another datapoint"

if __name__ == '__main__':
    test_dynamic_dataset()
