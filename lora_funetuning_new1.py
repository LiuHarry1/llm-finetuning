from datasets import load_dataset

# Load your dataset
dataset = load_dataset('json', data_files='sample_dataset.json')

# Split into train and validation sets
train_dataset = dataset['train'].train_test_split(test_size=0.1)['train']
val_dataset = dataset['train'].train_test_split(test_size=0.1)['test']

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

model_name = r"C:\apps\ml_model\llama2-7b-chat-hf"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # For text generation
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Tokenize the dataset
def tokenize_function(examples):
    # Concatenate prompt and response
    dialogue = examples['prompt'] + tokenizer.sep_token + examples['response']
    return tokenizer(dialogue, truncation=True, padding=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenizer.pad_token = tokenizer.eos_token

# Fine-tune the model using the Trainer API
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

# Fine-tune the model
trainer.train()


# Save the model
model.save_pretrained("./fine_tuned_llama3_lora")

# Conversion to GGUF can be done using external tools (e.g., llama.cpp), which might require exporting the model to a format like `pt` or `bin`.
# Example with Hugging Face Transformers:
model.save_pretrained("./fine_tuned_llama3_lora", safe_serialization=True)
