from typing import Annotated

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_tavily import TavilySearch

# https://www.kaggle.com/code/ksmooi/langgraph-dynamic-chart-generator-agent-teams/notebook

load_dotenv()

# Define DuckDuckGo Search Tool
@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Perform a search using DuckDuckGo and return the results.

    Args:
        query (str): The search query to be executed.
        max_results (int, optional): The maximum number of results to return. Defaults to 5.

    Returns:
        str: The search results as a string.
    """
    search = DuckDuckGoSearchRun()  # Initialize DuckDuckGoSearchRun
    results = search.run(query)  # Use run method for search
    return str(results)


# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()


@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Execute Python code using a Python REPL (Read-Eval-Print Loop).

    Args:
        code (str): The Python code to execute.

    Returns:
        str: The result of the executed code or an error message if execution fails.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# Create graph and define agent nodes
def make_system_prompt(suffix: str) -> str:
    """Generate a system prompt for the AI assistant.

    Args:
        suffix (str): Additional context or instructions to append to the base system prompt.

    Returns:
        str: The complete system prompt.
    """
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command

llm = ChatTongyi(model="qwen-plus")


def get_next_node(last_message: BaseMessage, goto: str):
    """Determine the next node to transition to based on the last message.

    Args:
        last_message (BaseMessage): The last message in the conversation.
        goto (str): The default node to transition to if no final answer is found.

    Returns:
        str: The next node to transition to, or END if a final answer is found.
    """
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto


# Research agent and node
research_task = "You can only do research. You are working with a chart generator colleague."
research_agent = create_react_agent(llm, tools=[duckduckgo_search], prompt=make_system_prompt(research_task))


def research_node(state: MessagesState) -> Command[Literal["chart_node", END]]:
    """Execute the research node, which performs research using the DuckDuckGo search tool.

    Args:
        state (MessagesState): The current state of the conversation.

    Returns:
        Command: A command object containing the updated state and the next node to transition to.
    """
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_node")
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="research_node"
    )
    return Command(update={"messages": result["messages"]}, goto=goto)


# Chart generator agent and node
chart_task = """Create clear and visually appealing charts using seaborn and plotly. Follow these rules:
1. Add a title, labeled axes (with units), and a legend if needed.
2. Use `sns.set_context("notebook")` for readable text and themes like `sns.set_theme()` or `sns.set_style("whitegrid")`.
3. Use accessible color palettes like `sns.color_palette("husl")`.
4. Choose appropriate plots: `sns.lineplot()`, `sns.barplot()`, or `sns.heatmap()`.
5. Annotate key points (e.g., "Peak in 2020") for clarity.
6. Ensure the chart's width and display resolution is no wider than 1000px.
7. Display with `plt.show()`.
Goal: Produce accurate, engaging, and easy-to-interpret charts."""
chart_agent = create_react_agent(llm, [python_repl_tool], prompt=make_system_prompt(chart_task))


def chart_node(state: MessagesState) -> Command[Literal["research_node", END]]:
    """Execute the chart node, which generates charts using the Python REPL tool.

    Args:
        state (MessagesState): The current state of the conversation.

    Returns:
        Command: A command object containing the updated state and the next node to transition to.
    """
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "research_node")
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_node"
    )
    return Command(update={"messages": result["messages"]}, goto=goto)





def print_pretty(event):
    """Pretty-print the event messages for debugging or logging purposes.

    Args:
        event (dict): The event containing messages from the research or chart node.
    """
    # Check if the event contains 'research_node' or 'chart_node'
    for node_key in ["research_node", "chart_node"]:
        if node_key in event:
            messages = event[node_key].get("messages", [])
            print(f"{node_key}: [")
            for message in messages:
                # Extract message type (HumanMessage, AIMessage, etc.)
                message_type = message.__class__.__name__

                # Extract message content
                content = message.content
                if isinstance(content, list):
                    content = [item for item in content]  # Handle list content (e.g., AIMessage with tool use)
                elif isinstance(content, str):
                    content = f'"{content}"'  # Wrap string content in quotes

                # Extract additional fields
                additional_kwargs = message.additional_kwargs
                response_metadata = message.response_metadata
                message_id = message.id

                # Print the message in the desired format
                print(f"    {message_type}(")
                print(f"        content={content},")
                print(f"        additional_kwargs={additional_kwargs},")
                print(f"        response_metadata={response_metadata},")
                print(f"        id='{message_id}'")
                print("    ),")
            print("]")
            print("-" * 120)
            return

    print("No messages found in the event.")



# Define the graph
from langgraph.graph import StateGraph, START

workflow = StateGraph(MessagesState)
workflow.add_node("research_node", research_node)
workflow.add_node("chart_node", chart_node)

workflow.add_edge(START, "research_node")
graph = workflow.compile()

# Invoke the graph
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="First, get the USA's population data for the past 50 years. "
                "Then, create a line chart with annotations for significant events like economic recessions. "
                "Add a trendline using numpy.polyfit. "
                "Once you make the chart, finish."
            )
        ],
    },
    {"recursion_limit": 150},
)

# Print events using print_pretty
for event in events:
    print_pretty(event)