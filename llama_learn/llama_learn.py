from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaTokenizer, LlamaForCausalLM

model_name = r"C:\apps\ml_model\llama2-7b-chat-hf"
print("Starting to load tokenizer.")
tokenizer = LlamaTokenizer.from_pretrained(model_name)

print("Starting to load model.")
# Load model with default precision (32-bit, unless specified otherwise)
model = LlamaForCausalLM.from_pretrained(model_name)

# Print model configuration
print("Model Configuration:")
print(model.config)

# Print model summary
print("\nModel Summary:")
print(model)

# Alternatively, use the summary method if available
try:
    model_summary = model.summary()
    print(model_summary)
except AttributeError:
    print("Summary method not available for this model.")


# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # For text generation
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    # target_modules=["q_proj", "v_proj"]
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print model configuration
print("lora Model Configuration:")
print(model.config)

# Print model summary
print("\n loral Model Summary:")
print(model)

# Alternatively, use the summary method if available
try:
    model_summary = model.summary()
    print(model_summary)
except AttributeError:
    print("Summary method not available for this model.")
