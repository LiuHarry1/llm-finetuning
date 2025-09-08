from typing import TypedDict, List, Literal, Dict, Any, Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.constants import END, START
from langgraph.graph import StateGraph



@tool("code_reflection", return_direct=True)
def code_reflection(error_message: str, code: Optional[str] = None) -> str:
    """
    通过大语言模型反思代码执行错误的工具。

    参数:
        error_message (str): 执行代码时的报错信息。
        code (Optional[str]): 出错的代码（可选）。

    返回:
        str: 大语言模型的反思结果，包括原因分析和修复建议。
    """
    print("starting code_reflection")
    prompt = f"""
你是一个专业的编程助手。现在有一段代码执行报错，请你进行反思。
输出需要包含三部分：
1. 错误信息重述  
2. 可能的原因分析  
3. 修复建议  

错误信息: {error_message}

出错代码:
```python
{code if code else ""}
"""

    response = llm.invoke(prompt)
    return response.content



@tool
def python_repl(code: str) -> str:
    """
    Use this function to execute Python code and get the results.
    """
    repl = PythonREPL()
    try:
        print("Running the Python REPL tool")
        print(code)
        result = repl.run(code)
        print(result)
        return f"Result of code execution: {result}"
    except Exception as e:
        print (f"Failed to execute. Error: {e!r}")
        return f"Failed to execute. Error: {e!r}"


tools = [python_repl, code_reflection]
tools_by_name = {tool.name: tool for tool in tools}


llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    """
    Represents the state of the Graph.
    """
    user_input: str  # User’s task request
    messages: List  # Chat history (questions, responses, tool outputs)
    new_input: str  # Flag to check for new user input
    code: str  # Stores the generated Python code
    iterations: int  # Tracks the number of execution attempts
    final_response: List  # Stores the final response after execution


def tool_node(state: State):
    """Performs the tool call"""
    result = []
    messages = state["messages"]

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": messages + result}
MAX_ITERS = 3

def should_continue(state: State) -> Literal["environment", END]:
    """
    Decide if we should continue execution or stop.
    """
    if state["iterations"] >= MAX_ITERS:
        return END

    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "Action"
    return END


def llm_call(state: State) -> Dict[str, Any]:
    """Generates code, decides on tool usage, and processes results."""
    print("----- Calling LLM -----")
    messages = list(state["messages"])
    user_input = state["user_input"]
    iterations = state["iterations"]
    new_input = state["new_input"]

    # Add system prompt on first run
    if not messages:
        messages.append(SystemMessage(content="""You are a Python coding assistant with expertise in exploratory data analysis.
Use the python_repl tool to execute the code. 
If the Python REPL tool fails (the output starts with "Failed to execute"), 
then call the `code_reflection` tool with the error message and the code.

If an error occurs, resolve it and retry up to 3 times. 
Once execution succeeds, analyze the result and provide insights.
Always structure responses with: prefix, code block, result, and analysis.
"""))

    # Add new user request
    if new_input == "True":
        messages.append(
            HumanMessage(content=f"Complete this task: {user_input}. Use the Python REPL tool if needed.")
        )
        new_input = "False"

    # Call LLM with tools
    code_solution = llm_with_tools.invoke(messages)
    messages.append(code_solution)

    return {
        "messages": messages,
        "final_response": code_solution,
        "iterations": iterations + 1,
        "new_input": new_input,
    }

agent_builder = StateGraph(State)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "environment",
        END: END
    }
)
agent_builder.add_edge("environment", "llm_call")
agent = agent_builder.compile()

# print(agent.get_graph().draw_ascii())
# png_data = agent.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png_data)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    # Example user question
    # question = """生成一个 365 天的时间序列数据（随机温度），绘制趋势折线图。"""
    # question = """"生成 100 个 [0, 2π] 区间内的点，绘制 sin(x) 和 cos(x) 的折线图。"""
    question = """给我在openshift server 上shutdown aspen-announcement 这个service　下面的所有的pods"""

    events = agent.stream(
        {"user_input": question, "messages": [], "iterations": 0, "new_input": "True"},
        config,
        stream_mode="values",
    )

    for event in events:
        for message in event["messages"]:
            message.pretty_print()