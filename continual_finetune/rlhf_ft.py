from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import PeftModel
import torch
import random

# ✅ 加载原始LoRA微调后的模型（LLaMA3-8B）
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b-chat-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "path_to_your_lora_adapter")
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

# ✅ PPO 参数配置
config = PPOConfig(
    model_name=None,
    learning_rate=1e-5,
    batch_size=2,
    forward_batch_size=1,
    log_with=None,
)

ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer)

# ✅ 示例 prompt 与反馈
prompts = [
    "请简要解释一下客户在JIRA-123中的问题。",
    "用户在JIRA-456报告了错误，请生成一个可能的原因分析。",
]

def generate_rewards(prompt, response):
    # 简化版的“人类反馈”，你可以替换成：
    # - GPT-4打分器
    # - Rule-based 逻辑
    # - 用户点击/接受反馈等
    if "错误" in response:
        return torch.tensor(1.0)
    return torch.tensor(0.5)

# ✅ 微调迭代
for epoch in range(3):
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        response_ids = model.generate(
            inputs["input_ids"], max_new_tokens=64, do_sample=True, top_k=50
        )
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        reward = generate_rewards(prompt, response_text)

        # PPO更新
        ppo_trainer.step([prompt], [response_text], [reward])
        print(f"Prompt: {prompt}")
        print(f"Response: {response_text}")
        print(f"Reward: {reward.item()}")
