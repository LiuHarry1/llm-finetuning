import os
from mem0 import Memory


from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from mem0.configs.base import MemoryConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.graphs.configs import GraphStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

#https://docs.mem0.ai/open-source/graph_memory/overview

llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d"
)

embeder = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key = "sk-f256c03643e9491fb1ebc278dd958c2d")

# 1. 配置 Memory
config = MemoryConfig( llm = LlmConfig( provider="langchain_learn", config={"model":llm }, ),
    embedder = EmbedderConfig( provider = "langchain_learn", config= { "model":embeder} ),
    vector_store = VectorStoreConfig(provider = "qdrant",
                                     config={
                                         "host": "localhost",
                                         "port": 6333,
                                         "collection_name": "memory_vectors",
                                         "embedding_model_dims": 1536,
                                     }
                                     ),

    graph_store=  GraphStoreConfig(provider = "neo4j",
                                       config= {
                                        "url": "bolt://localhost:7687",
                                        "username": "neo4j",
                                        "password": "myhome1234"

                                        }
                ),

    )

m = Memory(config=config)

related_memories = m.search(query="What do you know about me?", user_id="alice")
print("related_memories",related_memories)


