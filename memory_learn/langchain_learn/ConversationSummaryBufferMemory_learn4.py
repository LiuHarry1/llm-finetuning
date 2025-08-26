# 安装依赖（如果还没安装）
# pip install langchain openai

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")

llm = ChatTongyi(model="qwen-plus", api_key=TONGYI_API_KEY)

# 设置对话记忆（摘要）
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=50,
    memory_key="history",  # 和 Prompt 里的 MessagesPlaceholder 对应
    return_messages=True
)

# 假设一个简单的长期记忆
long_term_memory = ["你喜欢猫", "你正在学习Python"]

def get_long_term_memory():
    # 在这里可以接数据库 / 向量库检索
    return long_term_memory

# 定义 Prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个友好的聊天机器人。"),
    MessagesPlaceholder(variable_name="history"),  # 这里放记忆
    MessagesPlaceholder(variable_name="long_term_memory"),  # 这里放记忆
    HumanMessagePromptTemplate.from_template("{input}")
])


# 用 RunnablePassthrough + 管道操作符来组链
chain = ( RunnablePassthrough.assign
          (history=lambda x: memory.load_memory_variables({})["history"],
            long_term_memory=lambda x: get_long_term_memory()
            ) | prompt | llm )

# 聊天循环
print("欢迎使用简单聊天机器人！输入 '退出' 来结束。")
while True:
    user_input = input("你: ")
    if user_input.lower() in ["退出", "quit", "exit"]:
        print("聊天结束，再见！")
        break

    # 在运行对话前打印当前 memory
    print("\n=== 对话前的 Memory ===")
    current_memory = memory.load_memory_variables({})
    print(f"History: {current_memory['history']}")
    print("=======================\n")

    # 执行链
    response = chain.invoke({"input": user_input})
    print(f"Bot: {response.content}")

    # 更新 memory
    memory.save_context({"input": user_input}, {"output": response.content})

    # 在对话后也打印 memory，查看更新
    print("\n=== 对话后的 Memory ===")
    updated_memory = memory.load_memory_variables({})
    print(f"History: {updated_memory['history']}")
    print("=======================\n")
