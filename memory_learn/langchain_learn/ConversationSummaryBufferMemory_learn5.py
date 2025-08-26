# 安装依赖
# pip install langchain openai faiss-cpu tiktoken
# Qdrant 向量库
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import DashScopeEmbeddings
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# 向量库相关
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. 加载环境变量
load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")

# 2. 初始化 LLM
llm = ChatTongyi(model="qwen-plus", api_key=TONGYI_API_KEY)

# 3. 短期记忆（对话历史）
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200,
    memory_key="history",
    return_messages=True
)

# 4. 构建长期记忆（向量库）
embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key = TONGYI_API_KEY)
# 本地 Qdrant 实例（默认 6333 端口），或者换成远程 URL + API key

from qdrant_client.models import VectorParams, Distance
# 连接本地 Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)

collection_name = "memory_vectors1"

# 如果 collection 不存在就新建
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# 初始化 Embedding
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=TONGYI_API_KEY
)

# 构建 Qdrant 向量库（注意，这里传的是 url / collection_name，不再传 client）
texts = [
    "我叫刘浩",
    "喜欢打游戏",
    "我失业了"
]
qdrant = Qdrant.from_texts(
    texts=texts,
    embedding=embeddings,
    url="http://localhost:6333",     # ✅ 用 url 代替 client
    collection_name=collection_name
)

def get_long_term_memory(query: str):
    """根据用户输入做语义检索，返回相关内容"""
    docs = qdrant.similarity_search(query, k=2)
    print(query,"retrieved docs for long term memory", docs)
    return [d.page_content for d in docs]


# 5. 定义 Prompt，包含短期 + 长期记忆
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个友好的聊天机器人。"),
    MessagesPlaceholder(variable_name="long_term_memory"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# 6. 用 Runnable 串起来
chain = (
    RunnablePassthrough.assign(
        history=lambda x: memory.load_memory_variables({})["history"],
        long_term_memory=lambda x: get_long_term_memory(x["input"])
    )
    | prompt
    | llm
)

# 7. 聊天循环
print("欢迎使用智能聊天机器人！输入 '退出' 来结束。")
while True:
    user_input = input("你: ")
    if user_input.lower() in ["退出", "quit", "exit"]:
        print("聊天结束，再见！")
        break

    # 打印当前 memory
    print("\n=== 对话前的 Memory ===")
    print(memory.load_memory_variables({}))
    print("=======================\n")

    # 执行链
    response = chain.invoke({"input": user_input})
    print(f"Bot: {response.content}")

    # 更新 memory
    memory.save_context({"input": user_input}, {"output": response.content})

    # 打印 memory
    print("\n=== 对话后的 Memory ===")
    print(memory.load_memory_variables({}))
    print("=======================\n")
