from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# 假设我们有不同来源的消息列表
context_messages = [
    SystemMessage(content="以下是用户资料：年龄：25，职业：工程师。")
]

history_messages = [
    HumanMessage(content="我喜欢打篮球。"),
    AIMessage(content="很棒！运动对健康有益。")
]

current_question = "你能根据我的资料和之前的对话，推荐一些周末活动吗？"

# 创建模板，包含多个占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个个性化的活动推荐助手。"),
    MessagesPlaceholder(variable_name="context"),
    MessagesPlaceholder(variable_name="dialogue_history"),
    ("human", "{current_question}"),
])

# 格式化模板
final_messages = prompt.format_messages(
    context=context_messages,
    dialogue_history=history_messages,
    current_question=current_question
)

# 查看最终构建的提示（在发送给API之前）
for message in final_messages:
    print(f"{message.type}: {message.content}\n")

# 然后可以将 final_messages 发送给 LLM