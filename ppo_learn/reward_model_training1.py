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


from transformers import DataCollatorWithPadding

class MyRewardDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        # features 是 dict 的 list，包含 input_ids_chosen, input_ids_rejected 等
        # 你要把这些分开 collate
        chosen = [{"input_ids": f["input_ids_chosen"], "attention_mask": f["attention_mask_chosen"]} for f in features]
        rejected = [{"input_ids": f["input_ids_rejected"], "attention_mask": f["attention_mask_rejected"]} for f in features]

        chosen_batch = self.data_collator(chosen)
        rejected_batch = self.data_collator(rejected)

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }


def preprocess(example):
    # example 是 dict，包含原始文本
    chosen = example["chosen"]
    rejected = example["rejected"]
    prompt = example.get("prompt", "")

    tokenized_chosen = tokenizer(prompt + chosen, truncation=True, padding="max_length", max_length=1024)
    tokenized_rejected = tokenizer(prompt + rejected, truncation=True, padding="max_length", max_length=1024)

    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)



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





trainer = RewardTrainer(
    model=model,
    # tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    data_collator=MyRewardDataCollator(tokenizer)
)

trainer.train()

trainer.save_model("./output/rm-llama3")
tokenizer.save_pretrained("./output/rm-llama3")