from pydra import Config
import pydra
import os
import json

class CheckConfig(Config):
    def __init__(self):
        self.level = 1
        self.input_dir = f"reasoning_traces/level_{self.level}/eval"

@pydra.main(base=CheckConfig)
def main(config: CheckConfig):
    print(f"Checking JSON files in {config.input_dir}...")
    files = sorted(list(os.listdir(config.input_dir)))
    
    corrupted_files = []
    for file in files:
        file_path = os.path.join(config.input_dir, file)
        try:
            with open(file_path, "r") as f:
                try:
                    json.load(f)
                except json.JSONDecodeError as e:
                    print(f"\nCorrupted JSON file: {file}")
                    print(f"Error: {str(e)}")
                    corrupted_files.append(file)
        except Exception as e:
            print(f"\nError reading file {file}: {str(e)}")
            corrupted_files.append(file)
    
    if corrupted_files:
        print("\nSummary of corrupted files:")
        for file in corrupted_files:
            print(f"- {file}")
        print(f"\nTotal corrupted files: {len(corrupted_files)}")
    else:
        print("\nNo corrupted JSON files found!")

if __name__ == "__main__":
    main() 