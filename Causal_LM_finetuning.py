import json
from datasets import Dataset

# 创建因果语言模型数据集 (Causal LM)
causal_lm_data = [
    {"text": "今天的天气很好，我们去公园散步吧。"},
    {"text": "机器学习是一种人工智能方法，能够学习和改进。"},
    {"text": "自从他加入团队后，工作效率大大提高了。"}
]

# 保存到JSONL文件
with open("causal_lm_data.jsonl", "w", encoding="utf-8") as f:
    for entry in causal_lm_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

# 加载数据集
dataset = Dataset.from_json("causal_lm_data.jsonl")

# 打印数据集信息
print(dataset)

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset

# 加载数据集
dataset = Dataset.from_json("causal_lm_data.jsonl")

# 加载模型和标记器
model_name = "huggingface/llama-3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# 获取LoRA模型
peft_model = get_peft_model(model, lora_config)

# 定义数据处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 预处理数据集
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 准备训练参数
training_args = TrainingArguments(
    output_dir="./results_causal_lm",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    evaluation_strategy="epoch"
)

# 准备 Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset  # 这里简单使用同一个数据集作为示例
)

# 开始训练
trainer.train()
