import json
from datasets import Dataset

# 创建指令调优数据集 (Instruction Tuning)
instruction_tuning_data = [
    {"instruction": "翻译以下句子成英文：你好世界。", "response": "Hello, World."},
    {"instruction": "总结以下段落的主要观点。", "response": "段落的主要观点是……"},
    {"instruction": "写一段关于人工智能的简短介绍。", "response": "人工智能是一种模拟人类智能的技术..."}
]

# 保存到JSONL文件
with open("instruction_tuning_data.jsonl", "w", encoding="utf-8") as f:
    for entry in instruction_tuning_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

# 加载数据集
dataset = Dataset.from_json("instruction_tuning_data.jsonl")

# 打印数据集信息
print(dataset)

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset

# 加载数据集
dataset = Dataset.from_json("instruction_tuning_data.jsonl")

# 加载模型和标记器
# model_name = "huggingface/llama-3"  # 替换为您的模型名称
model_name = r"C:\apps\ml_model\llama2-7b-chat-hf"
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
    inputs = examples["instruction"]
    targets = examples["response"]
    # 将输入和输出连接为一个字符串，中间使用特殊分隔符（如 <s> 或 </s>）
    model_inputs = tokenizer(inputs + " </s> " + targets, max_length=256, truncation=True, padding="max_length")
    return model_inputs

# 预处理数据集
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 准备训练参数
training_args = TrainingArguments(
    output_dir="./results_instruction_tuning",
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
