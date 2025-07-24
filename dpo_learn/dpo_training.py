import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# ----------- 配置模型和路径 -----------
model_name = "C:\\apps\\ml_model\\Llama-3.2-1B-Instruct"
output_dir = ".\\dpo_llama3_lora"

# ----------- 加载分词器 -----------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ----------- 加载模型并应用 LoRA -----------
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quant_config,
    device_map="auto",
    # trust_remote_code=True
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 根据模型结构定制
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)


# ----------- 加载偏好数据集 -----------
def load_preference_dataset(jsonl_path):
    import json
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


# dataset = load_preference_dataset(dataset_path)
dataset = load_dataset("C:\\apps\\ml_datasets\\hh-rlhf", split="train", revision="harmless-base")

# ----------- DPO Trainer 配置 -----------
training_args = DPOConfig(
    beta=0.1,  # 偏好对比温度
    max_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    output_dir=output_dir,
    report_to="none",
    remove_unused_columns=False,
    fp16=True,  # 适用于支持的 GPU
    padding_value=tokenizer.pad_token_id,  # ← 加上这一行
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 不用引用模型，使用自身输出计算 log-likelihood
    args=training_args,
    train_dataset=dataset,
    # tokenizer=tokenizer,
)

# ----------- 启动训练 -----------
trainer.train()