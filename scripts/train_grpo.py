import os
import torch
import pydra
from pydra import REQUIRED, Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
from src.dataset import construct_kernelbench_dataset
from src.prompt_constructor import prompt_generate_prompt_with_hardware_info_from_template, SYSTEM_PROMPT
from src.utils import measure_program_time, set_gpu_arch, read_file
from src.reward import reward_fn

class TrainingConfig(Config):
    def __init__(self):
        # Model configuration
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"# "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
        self.learning_rate = 1e-5
        self.max_tokens = 512
        
        # GRPO configuration
        self.num_generations = 2  # Number of generations per prompt (G in the paper)
        self.beta = 0.01  # KL coefficient
        self.temperature = 0.9
        
        # Training configuration
        self.num_epochs = 10
        self.batch_size = 1
        self.gradient_accumulation_steps = 1
        
        # Dataset configuration
        self.level = 1
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.dataset_path = f"data/kernelbench_level_{self.level}"
        
        # Hardware configuration
        self.gpu_arch = ["Hopper"]  # GPU architecture for kernel compilation
        self.gpu_name = "H100"
        
        # Output configuration
        self.output_dir = "runs/grpo_training"

@pydra.main(base=TrainingConfig)
def main(config: TrainingConfig):
    # Set up GPU architecture for kernel compilation
    set_gpu_arch(config.gpu_arch)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Try to load existing dataset
    if os.path.exists(config.dataset_path):
        print(f"Loading preprocessed dataset from {config.dataset_path}")
        dataset = Dataset.load_from_disk(config.dataset_path)
    else:
        # Load and process dataset
        print("Loading dataset...")
        train_problems = construct_kernelbench_dataset(config.level)
        
        # Prepare training data
        print("Preparing training data...")
        train_prompts = []
        ref_arch_srcs = []
        baseline_runtimes = []
        
        for problem in tqdm(train_problems, desc="Processing problems"):
            ref_arch_src = read_file(problem)
            baseline_stats = measure_program_time(problem, ref_arch_src)
            if baseline_stats is None:
                print(f"Skipping problem {problem} due to baseline measurement error")
                continue
            baseline_runtimes.append(baseline_stats["mean"])
            prompt = prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src, gpu_name=config.gpu_name)
            train_prompts.append([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
            ref_arch_srcs.append(ref_arch_src)
        
        dataset = Dataset.from_dict({
            "prompt": train_prompts, 
            "ref_arch_src": ref_arch_srcs, 
            "baseline_runtime": baseline_runtimes
        })
        
        # Save processed dataset
        print(f"Saving preprocessed dataset to {config.dataset_path}")
        dataset.save_to_disk(config.dataset_path)

    print(f"Dataset size: {len(dataset)}")

    # Create GRPO config
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_prompt_length=None,
        max_completion_length=config.max_tokens,
        num_generations=config.num_generations,
        temperature=config.temperature,
        beta=config.beta,
        bf16=True,
        # use_vllm=True
    )

    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=config.model_name,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.save_pretrained(os.path.join(config.output_dir, "final_model"))

if __name__ == "__main__":
    main() 