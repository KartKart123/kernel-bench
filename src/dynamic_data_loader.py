import os
import json
import random
import torch
from datasets import IterableDataset, Dataset

def create_dynamic_loader(path, train_prompts, ref_arch_srcs, levels, task_ids, remove_on_consumption=True, shuffle=True):
    os.makedirs(path, exist_ok=True)
    file_counter = 0
    is_original = set()
    for prompt, ref_src, level, task_id in zip(train_prompts, ref_arch_srcs, levels, task_ids):
        with open(os.path.join(path, f"{file_counter}.json"), "w") as f:
            json.dump({"prompt": prompt, "ref_arch_src": ref_src, "level": level, "task_id": task_id}, f)
        is_original.add(file_counter)
        file_counter += 1

    def add_datapoint(train_prompt, ref_arch_src, level, task_id, original=False):
        nonlocal file_counter
        with open(os.path.join(path, f"{file_counter}.json"), "w") as f:
            json.dump({"prompt": train_prompt, "ref_arch_src": ref_arch_src, "level": level, "task_id": task_id}, f)
        if original:
            is_original.add(file_counter)
        file_counter += 1

    def generator():
        while True:
            files = [f for f in os.listdir(path) if f.endswith('.json') and os.path.isfile(os.path.join(path, f))]
            if not files:
                return
            file = random.choice(files) if shuffle else min(files, key=lambda x: int(x.removesuffix('.json')))
            filename = os.path.join(path, file)
            with open(filename, 'r') as f:
                data = json.load(f)
            file_id = int(file.removesuffix('.json'))
            if remove_on_consumption and file_id not in is_original:
                os.remove(filename)

            yield data

    dataset = IterableDataset.from_generator(generator)
    dataset.add_datapoint = add_datapoint   
    return dataset

def test():
    train_prompts = ["prompt_1", "prompt_2", "prompt_3"]
    ref_arch_srcs = ["source_1", "source_2", "source_3"]
    levels = [1, 1, 1]
    task_ids = [0, 1, 2]
    test_path = "test_data"

    dataset = Dataset.from_dict({
        "prompt": train_prompts,
        "ref_arch_src": ref_arch_srcs,
        "level": levels,
        "task_id": task_ids,
    })

    for element in dataset:
        print(f"consumed {element}")
        if random.random() < 0.5:
            dataset.add_datapoint("prompt_x", "source_x", 1, 3)
            print("added a new one")
        input()

if __name__ == '__main__':
    test()
