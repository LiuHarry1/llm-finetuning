from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载指定子集（如果是整合包，可以过滤）
dataset_harmless = load_dataset("C:\\apps\\ml_datasets\\hh-rlhf", split="train", revision="harmless-base")
# dataset_redteam = load_dataset("C:\\apps\\ml_datasets\\hh-rlhf", split="train", revision="red-team-attempts")
model_name = "C:\\apps\\ml_model\\Llama-3.2-1B-Instruct"

# 合并数据
# dataset = dataset_harmless.concatenate(dataset_redteam)
dataset= dataset_harmless
print(f"Total samples: {len(dataset)}")
print(dataset[0])
print(dataset[0]["chosen"])
print(dataset[0]["rejected"])
# print(dataset[0]["prompt"])

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 处理 padding

def preprocess(example):
    # example 是 dict，包含原始文本
    chosen = example["chosen"]
    rejected = example["rejected"]
    prompt = example.get("prompt", "")

    tokenized_chosen = tokenizer(prompt + chosen, truncation=True, padding="max_length", max_length=256)
    tokenized_rejected = tokenizer(prompt + rejected, truncation=True, padding="max_length", max_length=256)

    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

dataset = dataset.map(preprocess)

print(dataset[0])
