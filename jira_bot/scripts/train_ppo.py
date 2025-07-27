from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载LoRA + base模型
base_model = AutoModelForCausalLM.from_pretrained("model/base_model", device_map="auto", load_in_8bit=True)
model = PeftModel.from_pretrained(base_model, "adapters/lora_supervised")

# PPO配置
config = PPOConfig(
    model_name=None,
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    log_with=None,
)

# 构建奖励函数（模拟或基于评分）
def compute_reward(prompt, response):
    return 1.0 if "关键词" in response else -1.0  # 可替换为 GPT-4 的评分函数

# PPO训练
ppo_trainer = PPOTrainer(config, model=model, tokenizer=AutoTokenizer.from_pretrained("model/base_model"))

# 加载训练数据
import json
data = [json.loads(l) for l in open("data/rl_feedback_data.jsonl")]
for item in data:
    reward = compute_reward(item["question"], item["response"])
    ppo_trainer.step([item["question"]], [item["response"]], [reward])

model.save_pretrained("adapters/lora_ppo")
