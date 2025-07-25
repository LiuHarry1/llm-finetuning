from datasets import load_dataset

# 加载指定子集（如果是整合包，可以过滤）
dataset_harmless = load_dataset("C:\\apps\\ml_datasets\\hh-rlhf", split="train", revision="harmless-base")
# dataset_redteam = load_dataset("C:\\apps\\ml_datasets\\hh-rlhf", split="train", revision="red-team-attempts")

# 合并数据
# dataset = dataset_harmless.concatenate(dataset_redteam)
dataset= dataset_harmless
print(f"Total samples: {len(dataset)}")
print(dataset[0])
print(dataset[0]["chosen"])
print(dataset[0]["rejected"])
# print(dataset[0]["prompt"])
