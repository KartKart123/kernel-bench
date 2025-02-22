from together import Together
from dotenv import load_dotenv
from pydra import Config
import os
import pydra
import json
from src.dataset import construct_kernelbench_dataset
from src.prompt_constructor import custom_prompt_generate_custom_cuda
from src.utils import read_file
from tqdm import tqdm
class GenerationConfig(Config):
    def __init__(self):
        load_dotenv()
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        self.model_name = "deepseek-ai/DeepSeek-R1"
        self.max_tokens = 16384
        self.temperature = 0.7
        self.num_generations = 16
        self.level = 1
        self.problem_ids = [99]
        self.output_dir = f"reasoning_traces/level_{self.level}"

@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    client = Together(api_key=config.api_key)
    print("Loading dataset...")
    train_problems = construct_kernelbench_dataset(config.level)
    if config.problem_ids:
        train_problems = [train_problems[i] for i in config.problem_ids]
        
    for problem in tqdm(train_problems):
        print(f"Generating for problem: {problem}")
        ref_arch_src = read_file(problem)
        problem_name = os.path.basename(problem)
        problem_id = int(problem_name.split("_")[0])
        prompt = custom_prompt_generate_custom_cuda(ref_arch_src)
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
            n = config.num_generations,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            logprobs=1
        )
        
        os.makedirs(config.output_dir, exist_ok=True)
        output_data = {
            "problem_id": problem_id,
            "problem_name": problem_name,
            "generations": [
                {
                    "content": choice.message.content,
                    "tokens": choice.logprobs.tokens,
                    "token_logprobs": choice.logprobs.token_logprobs
                }
                for choice in response.choices
            ]
        }
        
        output_path = os.path.join(config.output_dir, f"{problem_name}.json")
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    main()