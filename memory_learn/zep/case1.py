from zep_python.memory.client import Memory

# 初始化本地 Zep Memory
memory1 = Memory(server_url="http://127.0.0.1:8000")  # 本地部署

# 用户输入
user_input = "帮我写一段 Python 爬虫"

# 检索历史相关记忆
history = memory1.get_relevant(user_input, top_k=3)

# 结合历史生成 prompt
context = "\n".join([h['content'] for h in history])
prompt = f"历史记忆:\n{context}\n用户问题:\n{user_input}"

# 调用你的 Qwen 模型生成回答
response = llm.predict(prompt)
print("Agent:", response)

# 保存本次对话到本地记忆
memory1.add(user_input, response)
