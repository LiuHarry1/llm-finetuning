import traceback

from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.chat_models import ChatTongyi
from langchain_experimental.tools import PythonREPLTool

# from langchain_experimental.utilities import PythonREPL
from langchain_tavily import TavilySearch
from langchain_core.tools import tool



load_dotenv()

@tool
def python_repl(code: str) -> str:
    """
    Use this function to execute Python code and get the results.
    """
    repl = PythonREPLTool()
    try:
        print("Running the Python REPL tool")
        result = repl.run(code)
        return f"Result of code execution: {result}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Execution failed with error: {e!r}\nTraceback:\n{tb}"

debug_llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

def debug_python_code(error_and_code: str) -> str:
    """
    Analyze Python error message and suggest fixes
    """
    prompt = f"请帮我debug以下Python错误，并给出修改建议:\n{error_and_code}"
    return debug_llm.predict(prompt)

# 定义 Tavily 搜索工具
search = TavilySearch(max_results=2)

tools = [
    Tool(
        name="python_repl",
        func=python_repl,
        description="Execute Python code and get result. Input must be valid Python code."
    ),
    Tool(
        name="search",
        func=search.run,
        description="Search the internet for general information."
    ),
    Tool(
        name="debug",
        func=debug_python_code,
        description="Analyze Python error message and code to suggest fixes."
    )
]

# 初始化 LLM

llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

# 初始化 ReAct Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("==== AI Agent Ready ====")
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print(f"AI: {response}")
