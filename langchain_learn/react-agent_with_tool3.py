import re

from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.chat_models import ChatTongyi
from langchain_experimental.tools import PythonREPLTool
# from langchain_experimental.utilities import PythonREPL
from langchain.callbacks.manager import CallbackManager
from langchain_tavily import TavilySearch
from langchain.callbacks.base import BaseCallbackHandler

import matplotlib.pyplot as plt
import os
import uuid

load_dotenv()

# 工具定义
python_repl = PythonREPLTool()
search = TavilySearch()

llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-f256c03643e9491fb1ebc278dd958c2d",
    streaming=True
)

def sanitize_input(query: str) -> str:
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = re.sub(r"^(\s|`)*(?i:py)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    query = re.replace("plt.show()", "", query)
    return query

# 改造 Python 工具：自动保存图表
def python_repl_tool(code: str) -> str:
    try:
        img_filename = f"/tmp/{uuid.uuid4().hex}.png"
        exec_globals = {"plt": plt, "img_filename": img_filename, "os": os}
        code = sanitize_input(code)
        exec(code, exec_globals)
        if plt.get_fignums():
            plt.savefig(img_filename)
            plt.close()
            return f"图表已生成: {img_filename}"
        return "代码执行完成，无图表生成。"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Execution failed with error: {e!r}\nTraceback:\n{tb}"

tools = [
    Tool(
        name="python_repl",
        func=python_repl_tool,
        description="执行Python代码并返回结果。可以生成图表并保存为图片。"
    ),
    Tool(
        name="search",
        func=search.run,
        description="搜索网络信息。"
    ),
]

class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

callback_manager = CallbackManager([StreamHandler()])
llm.callbacks = callback_manager

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# ========================
# 用户用自然语言提问
# ========================
query = "生成一个365天的随机温度数据，并绘制趋势折线图"
response = agent.run(query)
print("\nfinal:", response)
