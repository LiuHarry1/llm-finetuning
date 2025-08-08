from datasets import load_dataset
import json
from tqdm import tqdm

# 加载数据集（只用train部分）
ds = load_dataset("Amod/mental_health_counseling_conversations")["train"]

output = []

for item in tqdm(ds):
    context = item["Context"]
    response = item["Response"]

    example = {
        "messages": [
            {"role": "system", "content": "You are a supportive mental health counselor."},
            {"role": "user", "content": context.strip()},
            {"role": "assistant", "content": response.strip()}
        ]
    }

    output.append(example)

import os

os.makedirs("data", exist_ok=True)

with open("data/train.json", "w", encoding="utf-8") as f:
    for line in output:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

