from datasets import load_dataset
import json
from tqdm import tqdm
import os


ds = load_dataset("Amod/mental_health_counseling_conversations")
train_dataset = ds["train"]
test_dataset = ds["test"]

print(len(test_dataset))