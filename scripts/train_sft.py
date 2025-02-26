import os
import pydra
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig, get_peft_config
from peft import LoraConfig, get_peft_model
from pydra import Config
from datasets import load_dataset
from src.utils import get_tokenizer

class FTConfig(Config):
    def __init__(self):
        # Model settings
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        self.max_length = 32768
        
        # Training settings
        self.batch_size = 1
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = True
        self.learning_rate = 1e-5
        self.num_epochs = 2
        self.warmup_steps = 10
        
        # Data settings
        self.train_file = "data/sft/kernelbench_sft_level_1.json"
        self.output_dir = "runs/sft"
        
        # Logging settings
        self.logging_steps = 1
        self.save_steps = 500
        
        # Checkpoint settings
        self.resume_from_checkpoint = None # "runs/sft/checkpoint-1000"

        self.full_finetune = True
        # LoRA configuration
        self.lora_r = 8
        self.lora_alpha = self.lora_r
        self.lora_dropout = 0.0
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "down_proj", "up_proj"
        ]

@pydra.main(base=FTConfig)
def main(config: SFTConfig):
    
    dataset = load_dataset("json", data_files=config.train_file)["train"]

    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        use_cache=False if config.gradient_checkpointing else True,
        #device_map="auto"
    )

    tokenizer = get_tokenizer(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    peft_config = None

    if not config.full_finetune:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        model_init_kwargs=model_kwargs,
        learning_rate=config.learning_rate,
        bf16=True,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        report_to=["wandb"],
        save_total_limit=3
    )

    trainer = SFTTrainer(
        model=config.model_name,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )
    
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
if __name__ == "__main__":
    main() 