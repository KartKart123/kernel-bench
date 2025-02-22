import pydra
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from pydra import Config
import wandb
from datasets import load_dataset
class SFTConfig(Config):
    def __init__(self):
        # Model settings
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        self.max_length = 16384
        
        # Training settings
        self.batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 1e-5
        self.num_epochs = 3
        self.warmup_steps = 0
        
        # Data settings
        self.train_file = "data/sft/kernelbench_sft_level_1.json"
        self.output_dir = "runs/sft"
        
        # Logging settings
        self.logging_steps = 10
        self.save_steps = 500
        
        # Checkpoint settings
        self.resume_from_checkpoint = None # "runs/sft/checkpoint-1000"

        self.full_finetune = False
        # LoRA configuration
        self.lora_r = 8
        self.lora_alpha = self.lora_r
        self.lora_dropout = 0.0
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "down_proj", "up_proj"
        ]

@pydra.main(base=SFTConfig)
def main(config: SFTConfig):
    wandb.init(project="kernelbench-sft")

    dataset = load_dataset("json", data_files=config.train_file)["train"]
    input()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if not config.full_finetune:
        model = get_peft_model(model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        logging_dir="logs",
        report_to="wandb",
        save_total_limit=3,
        resume_from_checkpoint=config.resume_from_checkpoint
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    trainer.save_model()
    
    wandb.finish()

if __name__ == "__main__":
    main() 