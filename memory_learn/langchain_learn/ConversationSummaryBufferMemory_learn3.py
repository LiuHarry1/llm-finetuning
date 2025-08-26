# 安装依赖（如果还没安装）
# pip install langchain openai
import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi

from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")

llm = ChatTongyi(model="qwen-plus", api_key=TONGYI_API_KEY)

# 设置对话记忆（摘要）
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=50,
    memory_key="history",  # 注意这里改为 "history"
    return_messages=True
)

# 定义 Prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个友好的聊天机器人。"),
    MessagesPlaceholder(variable_name="history"),  # 这里添加记忆占位符
    HumanMessagePromptTemplate.from_template("{input}")
])

# 创建 LLMChain
chat_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# 聊天循环
print("欢迎使用简单聊天机器人！输入 '退出' 来结束。")
while True:
    user_input = input("你: ")
    if user_input.lower() in ["退出", "quit", "exit"]:
        print("聊天结束，再见！")
        break

    # 在运行对话前打印当前memory
    print("\n=== 对话前的 Memory ===")
    current_memory = memory.load_memory_variables({})
    print(f"History: {current_memory['history']}")
    print("=======================\n")

    response = chat_chain.run(user_input)
    print(f"Bot: {response}")

    # 在对话后也打印memory，查看更新
    print("\n=== 对话后的 Memory ===")
    updated_memory = memory.load_memory_variables({})
    print(f"History: {updated_memory['history']}")
    print("=======================\n")