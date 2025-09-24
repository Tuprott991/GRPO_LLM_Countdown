import torch
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from reward import CountdownReward
from tqdm import tqdm
import pandas as pd

def load_model_for_eval(model_id, peft_model_path=None):
    """Loads the base model and optionally merges LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if peft_model_path:
        model = PeftModel.from_pretrained(model, peft_model_path)
        model = model.merge_and_unload() # Merge adapters for faster inference

    model.eval()
    return model, tokenizer

def evaluate_model(model, tokenizer, test_dataset):
    """Evaluates the model on the test dataset and computes metrics."""
    reward_calculator = CountdownReward()
    results = []

    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    for example in tqdm(test_dataset):
        prompt = f"User: Using the numbers {example['numbers']}, create an equation that equals {example['target']}.\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)
        
        completion = tokenizer.decode(outputs[inputs.input_ids.shape:], skip_special_tokens=True)
        
        # Calculate metrics
        reward = reward_calculator.calculate_reward(completion, example['numbers'], example['target'])
        
        expression = reward_calculator._parse_completion(completion)
        has_valid_format = expression is not None
        
        is_correct = False
        uses_valid_numbers = False
        if has_valid_format:
            uses_valid_numbers, _ = reward_calculator._check_number_usage(expression, example['numbers'])
            try:
                result = reward_calculator.aeval.eval(expression)
                if result == example['target']:
                    is_correct = True
            except Exception:
                pass

        results.append({
            "prompt": prompt,
            "completion": completion,
            "reward": reward,
            "is_correct": is_correct,
            "has_valid_format": has_valid_format,
            "uses_valid_numbers": uses_valid_numbers and has_valid_format,
        })

    return pd.DataFrame(results)

def main():
    base_model_id = "microsoft/phi-2"
    tuned_model_path = "../models/grpo_tuned_model/final"
    test_dataset = load_dataset("json", data_files="../data/test_dataset.json", split="train")

    # --- Evaluate Base Model ---
    print("Evaluating base model...")
    base_model, tokenizer = load_model_for_eval(base_model_id)
    base_results_df = evaluate_model(base_model, tokenizer, test_dataset)
    del base_model # Free up memory
    torch.cuda.empty_cache()

    # --- Evaluate GRPO-Tuned Model ---
    print("\nEvaluating GRPO-tuned model...")
    tuned_model, tokenizer = load_model_for_eval(base_model_id, peft_model_path=tuned_model_path)
    tuned_results_df = evaluate_model(tuned_model, tokenizer, test_dataset)
    del tuned_model
    torch.cuda.empty_cache()

    # --- Print Summary ---
    summary = {
        "Metric": ["Accuracy (%)", "Valid Format (%)", "Valid Numbers (%)", "Average Reward"],
        "Base Model": [
            base_results_df['is_correct'].mean() * 100,
            base_results_df['has_valid_format'].mean() * 100,
            base_results_df['uses_valid_numbers'].mean() * 100,
            base_results_df['reward'].mean()
        ],
        "GRPO-Tuned Model": [
            tuned_results_df['is_correct'].mean() * 100,
            tuned_results_df['has_valid_format'].mean() * 100,
            tuned_results_df['uses_valid_numbers'].mean() * 100,
            tuned_results_df['reward'].mean()
        ]
    }
    summary_df = pd.DataFrame(summary)
    print("\n--- Evaluation Summary ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()