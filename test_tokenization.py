from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# model_name = r"C:\apps\ml_model\llama2-7b-chat-hf"
model_name = r"C:\apps\ml_model\llama3-8b-instruction-hf\llama-3-8b-chat-hf-latest"
# model_name = r"C:\apps\ml_model\qwen2.7-7b"
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_eos_token=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_eos_token = True
print(tokenizer.eos_token_id)
print(tokenizer.eos_token)
print(tokenizer.pad_token)
# tokenizer.pad_token = tokenizer.eos_token

text = "here is it "
model_inputs = tokenizer(text)
# model_inputs = tokenizer(text, max_length=20, truncation=True, padding="max_length")

print(model_inputs)


