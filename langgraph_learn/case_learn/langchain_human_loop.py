import os

from dotenv import load_dotenv
from langchain.agents import initialize_agent

from langchain.agents import AgentType
from langchain_community.tools import HumanInputRun
from langchain_community.chat_models import ChatTongyi

load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")

llm = ChatTongyi(model="qwen-plus", api_key=TONGYI_API_KEY)

tools = [
    HumanInputRun()  # 专门用于人工输入
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

chat_history = []  # 空列表也行
agent({"input": "帮我生成一段话，但是你需要先问我一个问题再继续。", "chat_history": chat_history})
