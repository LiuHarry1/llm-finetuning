from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk

from datasets import Dataset

data = {
    "instruction": [
        "What is the capital of France?",
        "What is 2 + 2?",
        "what is alert 38?",
        "what is initial notification?",
        "what is announcement in aspen?",
        "when alert 33 is raised in aspen system?",
        "How do you greet someone in English?"
    ],
    "response": [
        "The capital of France is Paris.",
        "2 + 2 equals 4.",
        "Alert 38 is to alert user to review position pulled from upstream after locking date. it needs to go through maker/checker.",
        "initial notification is created when event is created and this event has positions in any processing unit.",
        "announcement is created when there is an event in cooperation.",
        "alert 33 is high value alert, which is raised when there is high value payments.",
        "You greet someone by saying 'Hello' in English."
    ]
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk("simple_dataset")



# Load the model and tokenizer
model_name = r"C:\apps\ml_model\llama2-7b-chat-hf"
print("starting to load tokenizer.")
tokenizer = LlamaTokenizer.from_pretrained(model_name)
print("starting to load model.")
# Load model with 8-bit precision
model = LlamaForCausalLM.from_pretrained(model_name)
print("Finished to load model")
# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # This is for language modeling
    r=16,  # LoRA attention dimension
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
)

print("Starting to get peft model")
# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

tokenizer.pad_token = tokenizer.eos_token

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token':'PAD'})
#     model.resize_token_embeddings(len(tokenizer))


print("get peft model")
# Load the dataset
dataset = load_from_disk("simple_dataset")

# Tokenize the dataset
def tokenize_function(examples):
    combined_texts = [
        instruction + " " + response
        for instruction, response in zip(examples["instruction"], examples["response"])
    ]
    tokenized_inputs = tokenizer(
        combined_texts,
        truncation=True,
        padding="max_length",  # Ensure consistent padding
        max_length=128  # Adjust max_length as needed
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama2-lora-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    # fp16=True,  # Enable mixed precision
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
print("Starting to training")
# Fine-tune the model with LoRA
trainer.train()

print("Finished to training")
# Save the fine-tuned model
trainer.save_model("llama2-lora-finetuned")
print("model saved ")
