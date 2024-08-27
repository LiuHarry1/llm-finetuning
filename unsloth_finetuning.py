import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode
import torch

import unsloth
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load model and tokenizer
model_name = r"C:\apps\ml_model\llama2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



# Load and preprocess data
train_data = unsloth.load_jsonl_data("train_data.jsonl")
eval_data = unsloth.load_jsonl_data("eval_data.jsonl")

# Preprocess function to tokenize inputs and outputs
def preprocess_function(example):
    input_text = example['input']
    output_text = example['output']
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = tokenizer(output_text, return_tensors="pt").input_ids
    return {"input_ids": input_ids, "labels": output_ids}

# Preprocess datasets
train_dataset = train_data.map(preprocess_function)
eval_dataset = eval_data.map(preprocess_function)

# Training arguments
training_args = unsloth.TrainingArguments(
    output_dir="./fine_tuned_llama3",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500
)

# Trainer
trainer = unsloth.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_llama3")

# Convert to GGUF format if needed
unsloth.convert_model_to_gguf("./fine_tuned_llama3", "./fine_tuned_llama3_gguf")
