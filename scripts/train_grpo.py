import os
import json
import torch
import pydra
from pydra import REQUIRED, Config
from transformers import AutoModelForCausalLM
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
from src.dataset import construct_kernelbench_dataset
from src.prompt_constructor import prompt_generate_prompt_with_hardware_info_from_template, SYSTEM_PROMPT
from src.utils import measure_program_time, set_gpu_arch, read_file
from src.reward import reward_fn
from peft import LoraConfig, get_peft_model

class TrainingConfig(Config):
    def __init__(self):
        # Model configuration
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"# "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
        self.learning_rate = 1e-5
        self.max_tokens = 512
        
        # GRPO configuration
        self.num_generations = 2  # Number of generations per prompt
        self.beta = 0.01  # KL coefficient
        self.temperature = 0.6
        
        # Training configuration
        self.num_epochs = 10
        self.batch_size = 1
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = True
        self.use_vllm = True
        self.vllm_gpu_memory_utilization = 0.7
        
        # Dataset configuration
        self.level = 1
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.dataset_dir = "data"
        self.dataset_path = f"{self.dataset_dir}/kernelbench_level_{self.level}.json"
        
        # Hardware configuration
        self.gpu_arch = ["Hopper"]  # GPU architecture for kernel compilation
        self.gpu_name = "H100"
        
        # Output configuration
        self.output_dir = "runs/grpo_training"
        
        self.full_finetune = True
        # LoRA configuration
        self.lora_r = 8
        self.lora_alpha = self.lora_r
        self.lora_dropout = 0.01
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "down_proj", "up_proj"
        ]

@pydra.main(base=TrainingConfig)
def main(config: TrainingConfig):
    # Set up GPU architecture for kernel compilation
    set_gpu_arch(config.gpu_arch)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.dataset_dir, exist_ok=True)
    # Try to load existing dataset
    if os.path.exists(config.dataset_path):
        print(f"Loading preprocessed dataset from {config.dataset_path}")
        with open(config.dataset_path, "r") as f:
            data = json.load(f)
    else:
        # Load and process dataset
        print("Loading dataset...")
        train_problems = construct_kernelbench_dataset(config.level)
        
        # Prepare training data
        print("Preparing training data...")
        ref_arch_srcs = []
        baseline_runtimes = []
        
        for problem in tqdm(train_problems, desc="Processing problems"):
            ref_arch_src = read_file(problem)
            baseline_stats = measure_program_time(problem, ref_arch_src)
            if baseline_stats is None:
                print(f"Skipping problem {problem} due to baseline measurement error")
                continue
            baseline_runtimes.append(baseline_stats["mean"])
            ref_arch_srcs.append(ref_arch_src)
        
        data = {
            "ref_arch_src": ref_arch_srcs,
            "baseline_runtime": baseline_runtimes
        }

        # Save processed dataset
        print(f"Saving preprocessed dataset to {config.dataset_path}")
        with open(config.dataset_path, "w") as f:
            json.dump(data, f)
        
    train_prompts = []
    for ref_arch_src in data["ref_arch_src"]:
        prompt = prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src, gpu_name=config.gpu_name)
        train_prompts.append([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])

    dataset = Dataset.from_dict({
        "prompt": train_prompts,
        "ref_arch_src": data["ref_arch_src"],
        "baseline_runtime": data["baseline_runtime"]
    })

    print(f"Dataset size: {len(dataset)}")

    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        use_cache=False if config.gradient_checkpointing else True,
    )

    # Create GRPO config
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        model_init_kwargs=model_kwargs,
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
        use_vllm=config.use_vllm,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization
    )

    if config.full_finetune:
        model = config.model_name
    else:
        # Create model and apply LoRA
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Create GRPO trainer with LoRA model
    trainer = GRPOTrainer(
        model=model,
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