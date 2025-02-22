from torch.utils.data import IterableDataset
import os
import json
import random

class DynamicLoader(IterableDataset):
    def __init__(self, path, train_prompts, ref_arch_srcs, baseline_runtimes, levels, task_ids, remove_on_consumption=True, shuffle=True):
        self.path = path
        self.remove_on_consumption = remove_on_consumption
        self.file_counter = 0
        self.shuffle = shuffle
        os.makedirs(path, exist_ok=True)
        self.load_initial_training_data(train_prompts, ref_arch_srcs, baseline_runtimes, levels, task_ids)

    def add_datapoint(self, train_prompt, ref_arch_src, baseline_runtime, level, task_id):
        file_id = self.file_counter
        self.file_counter += 1

        data = {
            "prompt": train_prompt,
            "ref_arch_src": ref_arch_src,
            "baseline_runtime": baseline_runtime,
            "level" : level,
            "task_id": task_id
        }

        file_name = f"{file_id}.json"
        file_path = os.path.join(self.path, file_name)
        with open(file_path, "w") as f:
            json.dump(data, f)

    def load_initial_training_data(self, train_prompts, ref_arch_srcs, baseline_runtimes, levels, task_ids):
        for train_prompt, ref_arch_src, baseline_runtime, level, task_id in zip(train_prompts, ref_arch_srcs, baseline_runtimes, levels, task_ids):
            self.add_datapoint(train_prompt, ref_arch_src, baseline_runtime, level, task_id)

    def __iter__(self):
        while True:
            files = [f for f in os.listdir(self.path) if f.endswith('.json') and os.path.isfile(os.path.join(self.path, f))]
            if len(files) == 0:
                return
            else:
                file = random.choice(files) if self.shuffle else min(files, key = lambda string : int(string.remove_suffix('.json')))
                filename = os.path.join(self.path, file)
                with open(filename, 'r') as f:
                    data = json.load(f)

                if self.remove_on_consumption:
                    os.remove(filename)

                yield {
                    "prompt": data["prompt"],
                    "ref_arch_src": data["ref_arch_src"],
                    "baseline_runtime": data["baseline_runtime"],
                    "task_id": data["task_id"],
                    "level" : data["level"]
                }

    def __len__(self):
        return len([f for f in os.listdir(self.path) if f.endswith('.json') and os.path.isfile(os.path.join(self.path, f))])

def test():
    train_prompts = ["prompt_1", "prompt_2", "prompt_3"]
    ref_arch_srcs = ["source_1", "source_2", "source_3"]
    baseline_runtimes = [10, 20, 30]
    levels = [1,1,1]
    task_ids = [0,1,2]

    test_path = "test_data"

    loader = DynamicLoader(test_path, train_prompts, ref_arch_srcs, baseline_runtimes, levels, task_ids)
    for element in loader:
        if random.random() < .5:
            loader.add_datapoint("prompt_x", "source_x", 10, 1, 3)
if __name__ == '__main__':
    test()
