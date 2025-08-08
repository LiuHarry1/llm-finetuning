from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

base_model = "meta-llama/Llama-3-8b-chat-hf"
adapter_path = "lora-llama3-mental-health"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, adapter_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "<|user|> 最近总是很焦虑睡不着觉怎么办？\n<|assistant|>"
# prompt = "### User:\n我最近感觉很焦虑，睡不好觉，怎么办？\n\n### Counselor:\n"
output = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
print(output[0]["generated_text"])
