from peft import PeftModel, get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("model/base_model", device_map="auto", load_in_8bit=True)
model = PeftModel.from_pretrained(base_model, "adapters/lora_ppo")  # 继续在 PPO 基础上微调

# 可复用前面的训练逻辑
# 新数据：data/new_qa_data.json
