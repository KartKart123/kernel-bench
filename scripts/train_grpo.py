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
from src.prompt_constructor import custom_prompt_generate_custom_cuda, SYSTEM_PROMPT
from src.utils import measure_program_time, set_gpu_arch, read_file, get_tokenizer
from src.reward import reward_fn, format_reward
from peft import LoraConfig, get_peft_model

class TrainingConfig(Config):
    def __init__(self):
        # Model configuration
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
        self.learning_rate = 1e-5
        self.max_tokens = 4096
        
        # GRPO configuration
        self.num_generations = 7  # Number of generations per prompt
        self.beta = 0.001  # KL coefficient
        self.temperature = 0.7
        
        # Training configuration
        self.num_epochs = 20
        self.batch_size = 1
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = True
        self.use_vllm = True
        self.vllm_gpu_memory_utilization = 0.7
        self.optim = "adamw_torch"

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
        prompt = custom_prompt_generate_custom_cuda(ref_arch_src)#, gpu_name=config.gpu_name)
        # train_prompts.append([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        train_prompts.append([{"role": "user", "content": prompt}])
    dataset = Dataset.from_dict({
        "prompt": train_prompts,
        "ref_arch_src": data["ref_arch_src"],
        "baseline_runtime": data["baseline_runtime"],
        "task_id": list(range(len(train_prompts))), # TODO: add support for level 2
    })

    print(f"Dataset size: {len(dataset)}")

    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        use_cache=False if config.gradient_checkpointing else True,
    )
    tokenizer = get_tokenizer(config.model_name)

    # Create GRPO config
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        model_init_kwargs=model_kwargs if config.full_finetune else None,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        max_prompt_length=None,
        max_completion_length=config.max_tokens,
        num_generations=config.num_generations,
        temperature=config.temperature,
        beta=config.beta,
        bf16=True,
        use_vllm=config.use_vllm,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        optim=config.optim,
        report_to=["wandb"],
        logging_steps=1,
        save_steps=10,
        save_total_limit=10
    )

    if config.full_finetune:
        model = config.model_name
    else:
        # Create model and apply LoRA
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
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
        reward_funcs=[lambda prompts, completions, ref_arch_src, baseline_runtime, task_id, **kwargs: reward_fn(trainer, prompts, completions, ref_arch_src, baseline_runtime, task_id, **kwargs), 
                      format_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    if trainer.accelerator.is_main_process:
        print("Saving model...")
        trainer.model.config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main() 