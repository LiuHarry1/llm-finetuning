from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import load_dataset
import torch

# 模型名称
# base_model = r"C:\apps\ml_model\llama3-8b-instruction-hf\llama-3-8b-chat-hf"  # 或本地路径
base_model_name   = r"C:\apps\ml_model\Llama-3.2-3B-Instruct"
checkpoint_dir = "lora-llama3-mental-health/checkpoint-300"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 必须设置 pad_token

base_model = AutoModelForCausalLM.from_pretrained( base_model_name , load_in_8bit=True, device_map="auto", torch_dtype=torch.float16,)

model = PeftModel.from_pretrained(base_model, checkpoint_dir)

# 加载 LoRA 权重

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# 加载预处理数据
dataset = load_dataset("json", data_files="data/train.json")["train"]

# 格式处理：拼接 prompt
def format_prompt(example):
    messages = example["messages"]
    prompt = ""
    for m in messages:
        if m["role"] == "user":
            prompt += f"<|user|> {m['content']}\n"
        elif m["role"] == "assistant":
            prompt += f"<|assistant|> {m['content']}\n"
    return {"text": prompt}

dataset = dataset.map(format_prompt)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="lora-llama3-mental-health",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    learning_rate=5e-5,
    # bf16=True,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,

)

# trainer.train(resume_from_checkpoint=True)
trainer.train()
model.save_pretrained("lora-llama3-mental-health")
tokenizer.save_pretrained("lora-llama3-mental-health")
