import pydra
import os
import json
from pydra import Config
class InspectConfig(Config):
    def __init__(self):
        self.path = "reasoning_traces/level_1/eval"
        self.indices = [12]

@pydra.main(base=InspectConfig)
def main(config: InspectConfig):
    for index in config.indices:
        ref_arch_path = None
        dataset = os.listdir(config.path)
        for file in dataset:
            if file.split("_")[0] == str(index):
                ref_arch_path = file
                break
        if ref_arch_path is None:
            print(f"No file found starting with index {index}")
            continue
        with open(os.path.join(config.path, ref_arch_path), "r") as f:
            data = json.load(f)
            problem_name = data["problem_name"]
            for i, generation in enumerate(data["generations"]):
                content = generation["content"]
                print(content)
                print("-" * 50)
                print("\n\n")
                print(f"Generation {i} of {problem_name}")
                if "eval_result" in generation:
                    print(generation["eval_result"])
                else:
                    print("No eval_result found")
                input()

if __name__ == "__main__":
    main()