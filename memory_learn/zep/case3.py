import os
from zep_cloud.client import AsyncZep
from zep_cloud import Message

#https://zep.us/en/home/spaces

# 初始化 Zep 客户端
zep = AsyncZep(api_key=os.getenv("ZEP_API_KEY"))

# 创建一个新的会话
session_id = "user123-session"
await zep.memory.create(session_id)

# 添加用户消息到记忆
user_message = "Hi, I'm looking for a sci-fi movie recommendation."
await zep.memory.add(session_id, messages=[Message(role="user", content=user_message)])

# 获取当前会话的上下文
memory = await zep.memory.get(session_id)
print("Current Context:", memory.context)

# 模拟 AI 响应
ai_response = "How about 'Dune'? It's a great sci-fi movie."
await zep.memory.add(session_id, messages=[Message(role="assistant", content=ai_response)])

# 获取更新后的上下文
updated_memory = await zep.memory.get(session_id)
print("Updated Context:", updated_memory.context)
