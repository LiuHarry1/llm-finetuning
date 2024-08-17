from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig

# Load the base LLaMA 2 model
model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Load the PEFT fine-tuned model
peft_model_path = "./llama2-lora-finetuned"  # Path to your fine-tuned model
peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

# Merge the LoRA weights into the base model
peft_model = peft_model.merge_and_unload()

peft_model.save_pretrained("./llama2-finetuned-combined")
tokenizer.save_pretrained("./llama2-finetuned-combined")


from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("./llama2-finetuned-combined")
tokenizer = LlamaTokenizer.from_pretrained("./llama2-finetuned-combined")

# Now you can use the model for inference
inputs = tokenizer("What is the capital of France?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

