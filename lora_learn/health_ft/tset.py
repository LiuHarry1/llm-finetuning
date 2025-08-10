import torch

print("💡 PyTorch 是否检测到 GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ 当前使用的 GPU:", torch.cuda.get_device_name(0))
    print("🚀 GPU 总显存: {:.2f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print("🔥 当前显存使用: {:.2f} GB".format(torch.cuda.memory_allocated(0) / 1024**3))
else:
    print("❌ 没有检测到 GPU，将使用 CPU")

import transformers
print(transformers.__version__)

from transformers import TrainingArguments
print(TrainingArguments.__module__)