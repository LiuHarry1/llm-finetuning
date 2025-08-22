from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory

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
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
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
conversation_runnable = RunnableWithMessageHistory(
    llm=llm,
    memory=short_term_memory,
    input_key="input",
    return_messages=False
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

# 调用 Runnable
output = conversation_runnable.invoke({
    "input": user_input,
    "short_term_memory": short_term_memory.moving_summary_buffer,
    "long_term_memory": long_term_text
})

print(output["output"])
