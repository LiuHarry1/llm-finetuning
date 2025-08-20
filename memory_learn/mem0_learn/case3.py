import os
from mem0 import Memory


from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from mem0.configs.base import MemoryConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d"
)

embeder = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key = "sk-f256c03643e9491fb1ebc278dd958c2d")

# 1. 配置 Memory
config = MemoryConfig( llm = LlmConfig( provider="langchain", config={"model":llm }, ),
    embedder = EmbedderConfig( provider = "langchain", config= { "model":embeder} ),
    vector_store = VectorStoreConfig(provider = "qdrant",
                                     config={
                                         "host": "localhost",
                                         "port": 6333,
                                         "collection_name": "memory_vectors",
                                         "embedding_model_dims": 1536,
                                     }
                                     ),

                )


m = Memory(config=config)

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
]

# Store inferred memories (default behavior)
result = m.add(messages, user_id="alice", metadata={"category": "movie_recommendations"})

messages = [ {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}]
# Store memories with agent and run context
result = m.add(messages, user_id="alice", agent_id="movie-assistant", run_id="session-001",
               metadata={"category": "movie_recommendations"})

print(result)
specific_memory = m.get(result["results"][0]['id'])
print("specific_memory",specific_memory)

history = m.history(memory_id=result["results"][0]['id'])
print("history", history)

# Store raw messages without inference
# result = m.add(messages, user_id="alice", metadata={"category": "movie_recommendations"}, infer=False)

# Get all memories
all_memories = m.get_all(user_id="alice")

# Get a single memory by ID

print(all_memories)

related_memories = m.search(query="What do you know about me?", user_id="alice")
print("related_memories",related_memories)

result = m.update(memory_id=related_memories["results"][0]['id'], data="I love India, it is my favorite country.")
print(result)


# # Delete a memory by id
# m.delete(memory_id="892db2ae-06d9-49e5-8b3e-585ef9b85b8e")
# # Delete all memories for a user
# m.delete_all(user_id="alice")
#
# m.reset() # Reset all memories

