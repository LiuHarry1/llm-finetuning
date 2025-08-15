from memoripy import MemoryManager, JSONStorage
from memoripy.implemented_models import OpenAIChatModel, OllamaEmbeddingModel
from memory_learn.memoripy_learn2 import QwenChatModel, QwenEmbeddingModel

api_key = "sk-f256c03643e9491fb1ebc278dd958c2d"

chat_model = QwenChatModel(api_key=api_key, model_name="qwen-plus")
embed_model = QwenEmbeddingModel(api_key=api_key, model_name="text-embedding-v1")

storage = JSONStorage("memory_history.json")

memory_manager = MemoryManager(chat_model, embed_model, storage=storage)

# 创建一条“长期记忆”交互
prompt = "I live in Shanghai and I love xiaolongbao."
response = "Great! Shanghai is famous for its delicious soup dumplings."
embedding = memory_manager.get_embedding(prompt + " " + response)
concepts = memory_manager.extract_concepts(prompt + " " + response)

# 手动添加长期记忆


# 添加多次模拟访问
for i in range(12):  # 超过阈值 10
    memory_manager.add_interaction(prompt, response, embedding, concepts)
    memory_manager.retrieve_relevant_interactions()

# 检查长期记忆
short_term, long_term = memory_manager.load_history()
print("短期记忆数量：", len(short_term))
print("长期记忆数量：", len(long_term))
print("长期记忆内容：", long_term)