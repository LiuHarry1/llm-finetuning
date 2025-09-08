import operator
from typing import TypedDict, List, Literal, Dict, Any, Annotated

from langchain_community.chat_models import ChatTongyi
from langchain.chat_models import init_chat_model

from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages


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


tools = [python_repl]
tools_by_name = {tool.name: tool for tool in tools}


llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    """
    Represents the state of the Graph.
    """
    user_input: str  # User’s task request
    # messages: List  # Chat history (questions, responses, tool outputs)
    # messages : Annotated[list, operator.add]
    messages:  Annotated[list, add_messages]
    new_input: str  # Flag to check for new user input
    code: str  # Stores the generated Python code
    iterations: int  # Tracks the number of execution attempts
    final_response: List  # Stores the final response after execution
    last_error: str  # 新增：记录执行失败的错误信息


def tool_execute_node(state: State):
    print("----- tool_execute_node-----")
    result = []
    messages = state["messages"]

    for tool_call in messages[-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

        if "ERROR" in observation or "SyntaxError" in observation or "ModuleNotFoundError" in observation \
                or "ConfigException" in observation or "Error" in observation:
            return {
                "messages":  result,
                "last_error": observation
            }

    return {"messages":  result, "last_error": ""}



def reflect_node(state: State) -> Dict[str, Any]:
    print("----- Reflecting on error -----")
    # messages = state["messages"]
    messages = []
    last_error = state.get("last_error", "")

    result = []

    if last_error:
        messages.append(
            HumanMessage(content=f"代码执行失败，错误信息：\n{last_error}\n\n请分析错误并修复代码。")
        )

    reflection = llm.invoke(state["messages"]+ messages)
    result.append(reflection)
    # messages.append(reflection)

    return {
        # "messages": state["messages"] ,
        "messages": messages + [reflection],
        # "final_response": reflection,
        # "iterations": state["iterations"] + 1,
        "last_error": ""  # 清空错误，避免无限循环
    }


MAX_ITERS = 3


def should_continue(state: State) -> Literal["Action", END]:
    if state["iterations"] >= MAX_ITERS:
        return END

    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "Action"

    return END

def after_code_execution(state: State) -> Literal["reflect", "generate"]:
    if state.get("last_error"):
        return "reflect"
    return "generate"


def generate(state: State) -> Dict[str, Any]:
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
If an error occurs, resolve it and retry up to 3 times. 
Once execution succeeds, analyze the result and provide insights.
Always structure responses with: prefix, code block, result, and analysis.

## instructions
1. don't simulated environment, all is real.

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

agent_builder.add_node("generate", generate)
agent_builder.add_node("environment", tool_execute_node)
agent_builder.add_node("reflect", reflect_node)

agent_builder.add_edge(START, "generate")

agent_builder.add_conditional_edges(
    "generate",
    should_continue,   # 保持原有逻辑
    {
        "Action": "environment",
        END: END
    }
)

agent_builder.add_conditional_edges(
    "environment",
    after_code_execution,
    {
        "reflect": "reflect",
        "generate": "generate"
    }
)

agent_builder.add_edge("environment", "generate")
agent_builder.add_edge("reflect", "generate")
agent = agent_builder.compile()


# print(agent.get_graph().draw_ascii())
png_data = agent.get_graph().draw_mermaid_png()
with open("graph1.png", "wb") as f:
    f.write(png_data)


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

    for index, event in enumerate(events):
        print(f"============{index}==============")
        for message in event["messages"]:
            message.pretty_print()