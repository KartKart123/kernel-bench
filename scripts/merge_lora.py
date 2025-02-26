import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from pydra import Config, main
from src.utils import get_tokenizer

class MergeConfig(Config):
    def __init__(self):
        # Model settings
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        
        # LoRA adapter checkpoint path
        self.lora_checkpoint = "runs/sft/checkpoint-395"
        
        # Output directory for merged model
        self.output_dir = "runs/merged_model"
        
        # Device to load model on
        self.device = "auto"

@main(base=MergeConfig)
def main(config: MergeConfig):
    print(f"Loading base model {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device
    )
    
    print(f"Loading LoRA adapter from {config.lora_checkpoint}...")
    model = PeftModel.from_pretrained(
        model,
        config.lora_checkpoint
    )
    
    print("Merging LoRA adapter with base model...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    
    # Save tokenizer alongside the model
    tokenizer = get_tokenizer(config.model_name)
    tokenizer.save_pretrained(config.output_dir)
    
    print("Done! The merged model has been saved.")

if __name__ == "__main__":
    main() 