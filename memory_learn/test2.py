import os
from openai import OpenAI

# -----------------------------
# 初始化 Qwen 客户端
# -----------------------------
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# -----------------------------
# 聊天记忆类
# -----------------------------
class ChatMemory:
    def __init__(self, model_name="qwen-plus", token_limit=200):
        self.model_name = model_name
        self.token_limit = token_limit
        self.chat_history = []  # 保存完整聊天
        self.summary = ""       # 压缩后的摘要

    def _estimate_tokens(self, text: str) -> int:
        # 简单估算：1 token ≈ 4 字符
        return len(text) // 4

    def add_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})
        # 超过 token 限制时更新摘要
        total_tokens = self._estimate_tokens(self._full_text())
        if total_tokens > self.token_limit:
            self._update_summary()

    def _full_text(self):
        # 合并摘要和当前历史
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history])
        return self.summary + "\n" + history_text if self.summary else history_text

    def _update_summary(self):
        prompt = f"The following is a conversation between the user and assistant. Write a concise summary about the contents of this conversation.\n{self._full_text()}"
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        self.summary = completion.choices[0].message.content
        # 压缩完历史，只保留最后几条消息
        self.chat_history = self.chat_history[-2:]

    def get_memory(self):
        # 返回摘要 + 当前聊天
        return self.summary + "\n" + "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history])

# -----------------------------
# 示例使用
# -----------------------------
memory = ChatMemory(token_limit=60)  # token 限制为 100

# 添加初始聊天
memory.add_message("user", "What is LlamaIndex?")
memory.add_message("assistant", "LlamaIndex is the leading data framework for building LLM applications.")
memory.add_message("user", "Can you give me some more details?")
memory.add_message("assistant", "LlamaIndex is a framework for building context-augmented LLM applications...")

print("=== 当前记忆 ===")
print(memory.get_memory())

# 添加新消息
memory.add_message("user", "How can I use LlamaIndex to build a chatbot?")

print("\n=== 更新后记忆 ===")
print(memory.get_memory())
