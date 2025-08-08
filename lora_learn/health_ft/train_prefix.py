from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, PeftModel, PrefixTuningConfig, TaskType
from datasets import load_dataset
import torch

# 模型和数据设置
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16
)

# 构建 prefix-tuning 配置
peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=30,  # prefix 长度
    encoder_hidden_size=128,  # 小前缀编码器维度
    prefix_projection=True
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 加载并格式化数据集
dataset = load_dataset("Amod/mental_health_counseling_conversations")

def format(example):
    return {
        "text": f"### User:\n{example['context']}\n\n### Counselor:\n{example['response']}"
    }

dataset = dataset["train"].map(format)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

# 训练参数
training_args = TrainingArguments(
    output_dir="output_prefix",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
