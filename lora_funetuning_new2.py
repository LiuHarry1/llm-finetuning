

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# model_name = r"C:\apps\ml_model\llama2-7b-chat-hf"
model_name = "/Users/harry/Documents/apps/ml/llama-2-7b-chat"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)
print(tokenizer.eos_token)
tokenizer.pad_token = tokenizer.eos_token

simple_sentence = "This is a sentence."
token_ids = tokenizer(simple_sentence)

print(token_ids)