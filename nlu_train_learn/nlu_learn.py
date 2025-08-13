import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel, AdamW, get_scheduler
from datasets import load_dataset, load_metric
from tqdm import tqdm

# 1. 加载ATIS数据集
#https://huggingface.co/datasets/tuetschek/atis/viewer?views%5B%5D=train&row=19
dataset = load_dataset("atis")

# 2. 准备Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# 3. 标签映射（意图和槽位）
intent_labels = dataset['train'].features['label'].names  # 预设意图类别
slot_labels = dataset['train'].features['slots'].feature.names  # 预设槽位类别（包括O）

intent_label2id = {label: i for i, label in enumerate(intent_labels)}
slot_label2id = {label: i for i, label in enumerate(slot_labels)}


# 4. 数据预处理函数，做tokenizer编码和标签对齐
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding='max_length',
                                 max_length=50)

    labels_slots = []
    for i, slot_seq in enumerate(examples["slots"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # padding token
            elif word_idx != previous_word_idx:
                label_ids.append(slot_label2id[slot_seq[word_idx]])
            else:
                # 对子词做处理，这里简单用同一个标签，也可用BIO的I-标签
                label_ids.append(slot_label2id[slot_seq[word_idx]])
            previous_word_idx = word_idx
        labels_slots.append(label_ids)

    tokenized_inputs["labels_slots"] = labels_slots
    tokenized_inputs["labels_intent"] = [intent_label2id[label] for label in examples["label"]]
    return tokenized_inputs


# 5. 处理数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 6. 创建DataLoader
train_dataset = tokenized_datasets["train"].with_format("torch")
val_dataset = tokenized_datasets["validation"].with_format("torch")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


# 7. 定义模型
class BertForIntentAndSlot(nn.Module):
    def __init__(self, pretrained_model_name, num_intent_labels, num_slot_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        self.intent_classifier = nn.Linear(hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(hidden_size, num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                intent_labels=None, slot_labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = None
        if intent_labels is not None and slot_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits, intent_labels)
            slot_loss = loss_fct(slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1))
            loss = intent_loss + slot_loss
        return loss, intent_logits, slot_logits


# 8. 初始化模型
num_intent_labels = len(intent_labels)
num_slot_labels = len(slot_labels)

model = BertForIntentAndSlot('bert-base-uncased', num_intent_labels, num_slot_labels)

# 9. 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3  # 训练3个epoch
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 10. 训练和验证循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        intent_labels = batch["labels_intent"].to(device)
        slot_labels = batch["labels_slots"].to(device)

        loss, intent_logits, slot_logits = model(input_ids, attention_mask, token_type_ids, intent_labels, slot_labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss += loss.item()
    print(f"Epoch {epoch + 1} Train loss: {train_loss / len(train_loader):.4f}")

# 11. 简单验证示范（可扩展）
model.eval()
correct_intent = 0
total_intent = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        intent_labels = batch["labels_intent"].to(device)

        _, intent_logits, _ = model(input_ids, attention_mask, token_type_ids)
        preds = torch.argmax(intent_logits, dim=1)
        correct_intent += (preds == intent_labels).sum().item()
        total_intent += intent_labels.size(0)

print(f"Validation Intent Accuracy: {correct_intent / total_intent:.4f}")
