import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from reward import CountdownReward
import re

def main():
    # --- 1. Configuration ---
    model_id = "microsoft/phi-2"
    
    # PEFT/LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Quantization configuration for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # GRPO training configuration
    training_args = GRPOConfig(
        output_dir="../models/grpo_tuned_model",
        per_device_train_batch_size=1, # Each prompt is one "batch" item
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        num_generations=8, # Group size (G)
        logging_steps=10,
        save_steps=100,
        report_to="none", # Set to "wandb" for experiment tracking
        max_prompt_length=128,
        max_completion_length=256,
        remove_unused_columns=False,
        bf16=True, # Use bfloat16 for performance
    )

    # --- 2. Load Model and Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager" # Use flash_attention_2 for speed if available 
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False # Required for gradient checkpointing

    # --- 3. Load Dataset and Preprocess ---
    dataset = load_dataset("json", data_files="../data/train_dataset.json", split="train")

    def format_prompt(example):
        prompt = f"User: Using the numbers {example['numbers']}, create an equation that equals {example['target']}.\nAssistant:"
        return {"prompt": prompt}

    dataset = dataset.map(format_prompt)

    # --- 4. Define Reward Function ---
    reward_calculator = CountdownReward()

    def get_rewards(completions, prompts_data):
        rewards = []
        # TRL passes prompts as a list of dictionaries, we need to parse them back
        parsed_prompts = []
        for p_str in prompts_data:
            numbers_str = re.search(r'\[(.*?)\]', p_str).group(1)
            target_str = re.search(r'equals (\d+)', p_str).group(1)
            parsed_prompts.append({
                "numbers": [int(n) for n in numbers_str.split(',')],
                "target": int(target_str)
            })

        for i, completion in enumerate(completions):
            prompt_info = parsed_prompts[i]
            reward = reward_calculator.calculate_reward(
                completion, 
                prompt_info['numbers'], 
                prompt_info['target']
            )
            rewards.append(torch.tensor(reward, dtype=torch.float32))
        return rewards

    # --- 5. Initialize and Run Trainer ---
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=get_rewards,
    )

    print("Starting GRPO training...")
    trainer.train()

    # --- 6. Save Final Model ---
    final_model_path = "../models/grpo_tuned_model/final"
    trainer.save_model(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()