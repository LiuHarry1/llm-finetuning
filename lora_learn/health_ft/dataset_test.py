import math
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, EvalPrediction, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')


base_model  = r"C:\apps\ml_model\Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 必须设置 pad_token
# =========================
# 数据集拆分
# =========================
dataset = load_dataset("json", data_files="data/train.json")["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 格式处理
def format_prompt(example):
    messages = example["messages"]
    prompt = ""
    for m in messages:
        if m["role"] == "user":
            prompt += f"User: {m['content']}\n"
        elif m["role"] == "assistant":
            prompt += f"Assistant: {m['content']}\n"
    return {"text": prompt}


dataset = dataset.map(format_prompt)

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        # padding=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

print("Pad token id:", tokenizer.pad_token_id)

# 随便取一个样本的 input_ids
sample = tokenized_dataset["train"][0]["input_ids"]
print("Sample input_ids:", sample)

# 检查 sample 中是否包含 pad_token_id
if tokenizer.pad_token_id in sample:
    print("样本中包含 pad_token_id，说明正确加了 padding。")
else:
    print("样本中没有 pad_token_id，可能没正确加 padding。")

