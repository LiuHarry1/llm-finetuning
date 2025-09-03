from langchain_community.chat_models import ChatOllama
# from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain.chat_models import init_chat_model

# 初始化 Ollama 的 Llama3 模型
llm = init_chat_model(
    model="llama3.1:8b",          # 模型名直接写 ollama list 里看到的
    model_provider="ollama",      # 手动指定 provider
    model_kwargs={"temperature": 0}
)

resp = llm.invoke("hello")
print(resp.content)

llm = ChatOllama(model="llama3.1:8b", temperature=0.8)  # 确认名字和 ollama list 一致
resp = llm.invoke("Tell me about Mars")
print(resp.content)


embeddings = OllamaEmbeddings(model="llama3.1:8b")
vector = embeddings.embed_query("The quick brown fox jumps over the lazy dog")
print(len(vector), vector[:10])