from langchain_community.chat_models import ChatTongyi
from langchain.chains import ConversationChain
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
# 短期记忆（摘要形式）
# -----------------------
short_term_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=150,
    memory_key="short_term_memory",
    input_key="input",
    return_messages=False,  # 只返回文本摘要
)

# -----------------------
# 长期记忆（向量数据库）
# -----------------------
embeddings = DashScopeEmbeddings(model="text-embedding-v2",
                                 dashscope_api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# -----------------------
# Prompt 模板
# -----------------------
prompt_template = """
你是一个贴心的 AI 助手。

这是你目前的记忆：
- 短期记忆（最近的对话总结）: {short_term_memory}
- 长期记忆（用户的重要事实）: {long_term_memory}

请结合以上记忆和用户输入，给出合理的回答。

用户输入: {input}
"""

chat_prompt = ChatPromptTemplate.from_template(prompt_template)

# -----------------------
# ConversationChain
# -----------------------
conversation = ConversationChain(
    llm=llm,
    memory=short_term_memory,  # 只用短期记忆管理上下文
    prompt=chat_prompt,
    verbose=True,
)

# -----------------------
# 辅助函数：提取长期事实并存储
# -----------------------
def extract_and_store_long_term_facts(new_text: str):
    """调用 LLM 提取长期记忆并存入向量数据库"""
    fact_prompt = f"""
请从以下对话中提取适合长期保存的关键信息（如用户的身份、兴趣、习惯、喜好等），
用简洁的句子逐条列出：
{new_text}
"""
    facts = llm.predict(fact_prompt)
    print("\n[自动提取的长期事实]")
    print(facts)

    for line in facts.split("\n"):
        line = line.strip()
        if line:
            vectorstore.add_documents([Document(page_content=line, metadata={"source": "long_term_fact"})])

# -----------------------
# 模拟对话
# -----------------------
print("=== 开始对话 ===")
print(conversation.predict(input="你好，我叫小明"))
print(conversation.predict(input="我喜欢打篮球和编程"))
print(conversation.predict(input="昨天我在北京三里屯看了一部电影，非常好看"))
print(conversation.predict(input="你能记住我吗？"))

# -----------------------
# 自动把最后一轮对话提炼后存入长期记忆
# -----------------------
new_text = short_term_memory.moving_summary_buffer  # 使用摘要文本
extract_and_store_long_term_facts(new_text)

# -----------------------
# 检索长期记忆
# -----------------------
query = "小明住在哪里？他喜欢什么？"
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents(query)
print("\n=== 长期记忆检索结果 ===")
for d in docs:
    print(d.page_content)

# -----------------------
# 调试短期记忆
# -----------------------
print("\n=== 短期摘要 ===")
print(short_term_memory.moving_summary_buffer)
