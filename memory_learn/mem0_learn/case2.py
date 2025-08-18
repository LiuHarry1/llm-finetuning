from mem0 import Memory
import dashscope
import os

from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig


# 用 DashScope 的嵌入模型替代 OpenAI
def qwen_embedding(text: str):
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v2",
        input=text
    )
    return resp.output["embeddings"][0]["embedding"]


config = MemoryConfig(
    embedder= EmbedderConfig(),
    vector_store={
        "provider": "faiss",
        "config": {"collection_name": "qwen_memory"}
    },
    llm={
        "provider": "qwen",
        "config": {"api_key": "sk-f256c03643e9491fb1ebc278dd958c2d"}
    },
    history_db_path="local_history.sqlite",

)



# 初始化 Mem0 并覆盖默认的嵌入函数
m = Memory(config=config)





# 后续操作无需 OpenAI
m.add("我喜欢用 Qwen 模型编程", user_id="user123")
related = m.search("用户喜欢什么技术？", user_id="user123")
print(related)