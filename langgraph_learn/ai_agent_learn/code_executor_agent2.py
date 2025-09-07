from typing import TypedDict, List, Literal

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.constants import END, START
from langgraph.graph import StateGraph

#https://medium.com/@mariumaslam499/build-your-own-ai-coding-agent-with-langgraph-040644343e73
#https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/#code-solution

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
    except BaseException as e:
        return f"Failed to execute. Error: {e!r}"
    return f"Result of code execution: {result}"

tools = [python_repl]
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
max_iters = 3

def should_continue(state: State) -> Literal["environment", END]:
    """
    Decide if we should continue execution or stop.
    """
    messages = state["messages"]
    last_message = messages[-1]
    iterations = state["iterations"]

    if iterations > max_iters:
        return END  # Stop execution if max iterations are reached
    if last_message.tool_calls:
        return "Action"  # Continue execution if the LLM made a tool call
    return END


def llm_call(state: State):
    """
    The LLM agent node. Generates code, calls tools, and analyzes results.
    """
    print("----- Calling LLM -----")
    messages = state["messages"]
    user_input = state["user_input"]
    iterations = state["iterations"]
    new_input = state["new_input"]

    if len(messages) == 0:
        messages += [
            SystemMessage(content="""You are a Python coding assistant with expertise in exploratory data analysis.
            Use the python_repl tool to execute the code. If an error occurs, resolve it and retry up to 3 times.
            Once execution succeeds, analyze the result and provide insights.
            Structure responses with a prefix, code block, result, and analysis."""
                          )
        ]

    if new_input == "True":
        messages += [
            HumanMessage(
                content=f"The user wants to complete this task: {user_input}. Use the Python REPL tool to complete the task.")
        ]
        new_input = "False"

    code_solution = llm_with_tools.invoke(messages)
    messages += [(code_solution)]

    iterations += 1
    return {"messages": messages, "final_response": code_solution, "iterations": iterations, "new_input": new_input}

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

question = """From the dataset located at: "13100326.csv", 
Explore the relationship between glucose levels and glycated hemoglobin A1c (HbA1c) percentages within specific age groups. 
Identify any correlations or patterns between these two key indicators of blood sugar control."""

solution = agent.invoke({"user_input": question, "messages":[], "iterations": 0, "new_input": "True"})

print(solution)