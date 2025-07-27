from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel, get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import torch

# 1. 模型与Tokenizer加载
base_model_name = "meta-llama/Llama-3-8b"  # 替换为你本地或私有镜像的路径
lora_adapter_path = "./lora-checkpoint"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# 2. 加载已有LoRA adapter
model = PeftModel.from_pretrained(model, lora_adapter_path)
model.print_trainable_parameters()  # 查看可训练参数

# 3. 新的JIRA数据集
new_data = [
    {"question": "为什么审批流程失败？", "answer": "审批流程失败是因为缺少主管签字。"},
    {"question": "如何重置客户密码？", "answer": "请前往安全设置页面点击‘重置密码’按钮。"}
]

def format_prompt(example):
    return f"### 问题:\n{example['question']}\n\n### 回答:\n{example['answer']}"

# 转为 Huggingface Dataset
train_dataset = Dataset.from_list(new_data)
train_dataset = train_dataset.map(lambda e: {"text": format_prompt(e)})

# 4. Tokenization
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = train_dataset.map(tokenize_fn, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Training 配置
training_args = TrainingArguments(
    output_dir="./lora-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_total_limit=2,
    save_strategy="epoch",
    evaluation_strategy="no",
    bf16=True,
    report_to="none"
)

# 6. Trainer 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# 7. 保存新的LoRA adapter
model.save_pretrained("./lora-incremental")
tokenizer.save_pretrained("./lora-incremental")
