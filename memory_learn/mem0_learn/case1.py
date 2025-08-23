import os
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig

"""
https://zhuanlan.zhihu.com/p/1908242232361870846
"""
# 配置 LLM 客户端（以 OpenAI 为例）
from openai import OpenAI

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d"
)

embeder = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key = "sk-f256c03643e9491fb1ebc278dd958c2d")

# 1. 配置 Memory
config = MemoryConfig( llm = LlmConfig( provider="langchain_learn", config={"model":llm }, ),
    embedder = EmbedderConfig( provider = "langchain_learn", config= { "model":embeder} )
)

# 2. 初始化 Memory
memory = Memory(config =config )

memory.add("我喜欢用 Qwen 模型编程", user_id="user123")
related = memory.search("用户喜欢什么技术？", user_id="user123")
print(related)



