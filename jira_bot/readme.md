           ┌────────────┐
           │ 原始模型    │
           └────┬───────┘
                ↓
       LoRA 微调（QA数据）
                ↓
     adapter → adapters/lora_supervised
                ↓
        PPO 强化学习（用户反馈）
                ↓
     adapter → adapters/lora_ppo
                ↓
    再次 LoRA 微调（新Ticket数据）
                ↓
     adapter → adapters/lora_final
                ↓
       ⬇️部署为 RAG-Bot
