from datasets import load_dataset
import json
from tqdm import tqdm
import os

# 加载原始数据集
dataset = load_dataset("Amod/mental_health_counseling_conversations")["train"]

# 按 90%:10% 划分 train/test
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 定义一个函数，把数据转换成 LoRA 需要的格式
def convert_to_lora_format(ds):
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
    return output

train_output = convert_to_lora_format(split_dataset["train"])
test_output = convert_to_lora_format(split_dataset["test"])

# 保存文件
os.makedirs("data", exist_ok=True)

with open("data/train.json", "w", encoding="utf-8") as f:
    for line in train_output:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

with open("data/test.json", "w", encoding="utf-8") as f:
    for line in test_output:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
