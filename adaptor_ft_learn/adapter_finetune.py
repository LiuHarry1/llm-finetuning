from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, AdapterTrainer
from transformers import AdapterConfig

from datasets import load_dataset

# 加载数据
dataset = load_dataset("glue", "sst2")
dataset = dataset.map(lambda x: {"labels": x["label"]}, remove_columns=["label"])
dataset.set_format(type="torch", columns=["sentence", "labels"])

# 分词
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize_fn(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize_fn, batched=True)

train_ds = dataset["train"]
val_ds = dataset["validation"]

# 加载模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 添加 Adapter
adapter_name = "sst2_adapter"
config = AdapterConfig.load("pfeiffer")  # 或 "houlsby", "prefix_tuning"
model.add_adapter(adapter_name, config=config)
model.train_adapter(adapter_name)  # 冻结原始模型，只训练 adapter

# 训练参数
args = TrainingArguments(
    output_dir="./adapter_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
)

# 启动训练
trainer = AdapterTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {"acc": (p.predictions.argmax(-1) == p.label_ids).mean()},
)

trainer.train()
