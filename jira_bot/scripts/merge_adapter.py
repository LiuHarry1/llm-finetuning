from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("model/base_model", device_map="auto")
model = PeftModel.from_pretrained(base, "adapters/lora_ppo")
model = model.merge_and_unload()
model.save_pretrained("final_model")
