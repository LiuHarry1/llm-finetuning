from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma, Qdrant

from langchain.schema import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_qdrant import Qdrant

from langchain.schema import BaseChatMessageHistory
from qdrant_client import QdrantClient

# -----------------------
# 初始化 LLM
# -----------------------
llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

# -----------------------
# 短期记忆
# -----------------------
short_term_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=150,
    memory_key="short_term_memory",
    input_key="input",
    return_messages=False
)

# -----------------------
# 长期记忆（向量数据库）
# -----------------------
embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

client = QdrantClient(url="http://localhost:6333")  # 本地 Qdrant 服务


if "long_term_memory" not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name="long_term_memory",
        vectors_config={"size": 1536, "distance": "Cosine"}  # size 根据 embedding 维度改
    )

vectorstore = Qdrant(
    client=client,
    collection_name="long_term_memory",
    embeddings=embeddings
)

# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})




# -----------------------
# Prompt
# -----------------------
prompt_template = """
你是一个贴心的 AI 助手。
短期记忆: {short_term_memory}
长期记忆: {long_term_memory}

用户输入: {input}
"""

chat_prompt = ChatPromptTemplate.from_template(prompt_template)

# -----------------------
# RunnableWithMessageHistory
# -----------------------

from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

# 定义一个简单的 Runnable 包装 LLM
class LLMRunnable(Runnable):
    def invoke(self, inputs: dict) -> dict:
        prompt = chat_prompt.format(
            input=inputs["input"],
            short_term_memory=short_term_memory.moving_summary_buffer,
            long_term_memory=inputs.get("long_term_memory", "")
        )
        return {"output": llm(prompt)}

llm_runnable = LLMRunnable()




# 返回一个空 MessageHistory 或从 Memory 转换
conversation_runnable = RunnableWithMessageHistory(
    runnable=llm_runnable,
    get_session_history=lambda session: BaseChatMessageHistory(messages=[])
)


# -----------------------
# 辅助函数：获取长期记忆文本
# -----------------------
def get_long_term_memory_text(query):
    docs = retriever.get_relevant_documents(query)
    return "\n".join([d.page_content for d in docs])

# -----------------------
# 对话示例
# -----------------------
user_input = "你好，我叫小明"
long_term_text = get_long_term_memory_text(user_input)

session_id = "user_1"
# 调用 Runnable
output = conversation_runnable.invoke(
    {
        "input": user_input,
        "long_term_memory": long_term_text
    },
    {"configurable": {"session_id": session_id}}
)

print(output["output"])
