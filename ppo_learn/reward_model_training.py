from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import RewardTrainer, RewardConfig
import torch
from datasets import load_dataset, Dataset
# from trl.trainer import DataCollatorWithPadding  # 新增导入
from transformers import DataCollatorWithPadding

model_name = "C:\\apps\\ml_model\\Llama-3.2-1B-Instruct"
dataset = load_dataset("C:\\apps\\ml_datasets\\hh-rlhf", split="train", revision="harmless-base")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 处理 padding

# 自定义数据整理器
class RewardDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        # 添加奖励模型需要的特殊字段处理
        if "chosen_input_ids" in features[0]:
            batch["chosen_input_ids"] = torch.stack([torch.tensor(f["chosen_input_ids"]) for f in features])
            batch["chosen_attention_mask"] = torch.stack([torch.tensor(f["chosen_attention_mask"]) for f in features])
        if "rejected_input_ids" in features[0]:
            batch["rejected_input_ids"] = torch.stack([torch.tensor(f["rejected_input_ids"]) for f in features])
            batch["rejected_attention_mask"] = torch.stack([torch.tensor(f["rejected_attention_mask"]) for f in features])
        return batch

# 自定义奖励模型的数据整理器
def preprocess_function(examples):
    chosen = tokenizer(examples["chosen"], truncation=True, max_length=1024)
    rejected = tokenizer(examples["rejected"], truncation=True, max_length=1024)
    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"]
    }

dataset = dataset.map(preprocess_function, batched=True)
data_collator = RewardDataCollator(tokenizer=tokenizer, padding="max_length")


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_8bit=True,  # 节省显存
    torch_dtype=torch.float16,
    device_map="auto"
)

# 应用 LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)



training_args = RewardConfig(
    output_dir="./output/rm-llama3",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    max_length=1024,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    # bf16=True,  # 或根据硬件选择 fp16/bf16
    fp16= True
)

# data_collator = RewardDataCollatorWithPadding(
#     tokenizer=tokenizer,
#     max_length=1024,
#     padding="max_length"
# )

trainer = RewardTrainer(
    model=model,
    # tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    data_collator=data_collator,  # 添加data_collator参数
)

trainer.train()

trainer.save_model("./output/rm-llama3")
tokenizer.save_pretrained("./output/rm-llama3")