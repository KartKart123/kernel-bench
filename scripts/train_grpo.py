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
from src.reward import reward_fn, compute_format_reward
from peft import LoraConfig, get_peft_model

class TrainingConfig(Config):
    def __init__(self):
        self.seed = 11
        self.verbose = True
        # Model configuration
        self.model_name = "runs/merged_model" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" 
        self.learning_rate = 1e-5
        self.max_tokens = 16384

        # GRPO configuration
        self.num_generations = 7 # Number of generations per prompt
        self.beta = 0  # KL coefficient
        self.temperature = 0.7
        
        # Training configuration
        self.num_epochs = 10
        self.batch_size = 1
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = True
        self.use_vllm = True
        self.vllm_gpu_memory_utilization = 0.5
        self.optim = "adamw_torch"

        # Evaluation configuration
        self.do_eval = False
        self.per_device_eval_batch_size = self.batch_size
        self.eval_strategy = "steps" if self.do_eval else "no"
        self.eval_steps = 20

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
        self.response_output_dir = "outputs"

        self.full_finetune = True # LoRA is broken for now, it's probably because of https://github.com/vllm-project/vllm/issues/12961 
        # LoRA configuration
        self.lora_r = 8
        self.lora_alpha = self.lora_r
        self.lora_dropout = 0.0
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
        levels = []
        task_ids = []
        
        for i, problem in enumerate(tqdm(train_problems, desc="Processing problems")):
            ref_arch_src = read_file(problem)
            task_id = int(os.path.basename(problem).split('_')[0])
            baseline_stats = measure_program_time(problem, ref_arch_src, verbose=True)
            if baseline_stats is None:
                print(f"Skipping problem {problem} due to baseline measurement error")
                continue
            baseline_runtimes.append(baseline_stats["mean"])
            ref_arch_srcs.append(ref_arch_src)
            levels.append(config.level)
            task_ids.append(task_id)

        data = {
            "ref_arch_src": ref_arch_srcs,
            "baseline_runtime": baseline_runtimes,
            "level": levels,
            "task_id": task_ids
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
        "level": data["level"],
        "task_id": data["task_id"],
    })

    print(f"Dataset size: {len(dataset)}")

    # Split into train and eval
    split_dataset = dataset.train_test_split(test_size=0.01, seed=config.seed, shuffle=False)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    if config.verbose:
        print("Evaluation dataset indices:")
        print(eval_dataset["task_id"])

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
        save_steps=50,
        save_total_limit=3,
        do_eval=config.do_eval,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        seed=config.seed,
    )

    peft_config = None
    model = config.model_name

    if not config.full_finetune:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=False if config.gradient_checkpointing else True,
        )
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        # print("Compiling model")
        # model = torch.compile(model)

    # Create reward function that takes in trainer
    def compute_reward(prompts, completions, ref_arch_src, level, task_id, **kwargs):
        return reward_fn(prompts, completions, ref_arch_src, level, task_id, 
                         trainer, output_dir=config.response_output_dir, **kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_reward, compute_format_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    # Train model
    print("Starting training...")
    trainer.train()
    
    if trainer.accelerator.is_main_process:
        print("Saving model...")
        trainer.model.config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main() 