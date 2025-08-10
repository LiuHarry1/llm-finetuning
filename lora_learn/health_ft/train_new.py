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
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 用这里传入
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

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

# 计算 loss 和 PPL
def compute_metrics(eval_pred: EvalPrediction):

    loss = eval_pred.metrics["eval_loss"] if "eval_loss" in eval_pred.metrics else None
    if loss is not None:
        ppl = math.exp(loss) if loss < 20 else float("inf")  # 避免 loss 太大溢出
        return {"eval_loss": loss, "eval_ppl": ppl}
    else:
        return {}

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
        # padding="max_length",
        padding=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# =========================
# Trainer
# =========================
training_args = TrainingArguments(
    output_dir="lora-llama3-mental-health",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    learning_rate=5e-5,
    fp16=True,
    eval_strategy="epoch",  # 每个 epoch 评估一次
    # eval_strategy="steps", # 按 steps 评估
    # eval_steps=100,              # 每 200 步评估一次
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,  # 加上 metrics
)

trainer.train()

# 保存模型
model.save_pretrained("lora-llama3-mental-health")
tokenizer.save_pretrained("lora-llama3-mental-health")


# ====== 绘制 训练集 Loss + 验证集 Loss + 验证集 PPL ======
train_loss_values = []
train_steps = []
eval_loss_values = []
eval_steps = []

for log in trainer.state.log_history:
    if "loss" in log and "learning_rate" in log:
        train_loss_values.append(log["loss"])
        train_steps.append(log["step"])
    if "eval_loss" in log:
        eval_loss_values.append(log["eval_loss"])
        eval_steps.append(log.get("step", log.get("epoch", None)))

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(train_steps, train_loss_values, label="Train Loss", color='blue', alpha=0.6)
plt.plot(eval_steps, eval_loss_values, label="Eval Loss", marker='o', color='green')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig("train_eval_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()