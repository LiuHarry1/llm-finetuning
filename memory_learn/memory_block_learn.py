import os
from openai import OpenAI
from llama_index.core.memory import (
    StaticMemoryBlock,
    FactExtractionMemoryBlock,
    VectorMemoryBlock,
)



class FactExtractionMemory:
    def __init__(self, client, model="qwen-plus", max_facts=20):
        self.client = client
        self.model = model
        self.max_facts = max_facts
        self.facts = []  # 存储提取的事实

    def extract_facts(self, conversation):
        """
        conversation: [(role, content), ...] 格式的对话列表
        """
        # 把对话拼成文本
        conversation_text = "\n".join(f"{r}: {c}" for r, c in conversation)

        prompt = f"""
        从以下对话中提取尽可能精确的事实（例如姓名、年龄、地点、喜好等），
        每条事实保持简短，不超过一句话。
        最多保留 {self.max_facts} 条事实。

        对话内容：
        {conversation_text}

        输出格式：
        - 事实1
        - 事实2
        ...
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个从对话中提取关键信息的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # 解析 LLM 输出
        output = completion.choices[0].message.content.strip()
        extracted = [line.strip("- ").strip() for line in output.split("\n") if line.strip()]
        self.facts.extend(extracted)
        # 去重
        self.facts = list(dict.fromkeys(self.facts))
        # 限制最大数量
        self.facts = self.facts[:self.max_facts]

    def get_facts(self):
        return self.facts


# ==== 使用示例 ====
if __name__ == "__main__":
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    fact_memory = FactExtractionMemory(client)

    conversation = [
        ("user", "我叫张伟，今年30岁，在北京工作。"),
        ("assistant", "你喜欢什么运动？"),
        ("user", "我喜欢打篮球，也喜欢踢足球。"),
        ("user", "我家有两只猫。")
    ]

    fact_memory.extract_facts(conversation)
    print("提取的事实：")
    for fact in fact_memory.get_facts():
        print("-", fact)
