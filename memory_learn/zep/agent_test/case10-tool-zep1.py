import asyncio
import traceback
import uuid
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_community.chat_models import ChatTongyi
import os
from langchain_core.messages import AIMessage, SystemMessage, trim_messages, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from zep_cloud import AsyncZep, Message, Zep

class State(TypedDict):
    messages: Annotated[list, add_messages]
    first_name: str
    last_name: str
    thread_id: str
    user_name: str


@tool
def search_facts(state: State, query: str, limit: int = 5) -> list[str]:
    """Search for facts in all conversations had with a user.

    Args:
        state (State): The Agent's state.
        query (str): The search query.
        limit (int): The number of results to return. Defaults to 5.

    Returns:
        list: A list of facts that match the search query.
    """
    print("starting to call search facts method", state)
    search_result = zep.graph.search(
        user_id=state["user_name"], query=query, limit=limit, scope="edges"
    )
    edges =  search_result.edges

    return [edge.fact for edge in edges]


@tool
async def search_nodes(state: State, query: str, limit: int = 5) -> list[str]:
    """Search for nodes in all conversations had with a user.

    Args:
        state (State): The Agent's state.
        query (str): The search query.
        limit (int): The number of results to return. Defaults to 5.

    Returns:
        list: A list of node summaries for nodes that match the search query.
    """
    print("starting to call search node method", state)
    search_result = zep.graph.search(
        user_id=state["user_name"], query=query, limit=limit, scope="nodes"
    )
    nodes = search_result.nodes
    return [node.summary for node in nodes]

load_dotenv(override=True)
ZEP_API_KEY = os.getenv("ZEP_API_KEY")
# zep = AsyncZep(api_key=ZEP_API_KEY)
zep = Zep(api_key=ZEP_API_KEY)
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")

llm = ChatTongyi( model="qwen-plus", api_key=TONGYI_API_KEY)

tools = [search_facts, search_nodes]
tool_node = ToolNode(tools)
# llm = llm.bind_tools(tools)

async def chatbot(state: State):
    memory = zep.thread.get_user_context(state["thread_id"])

    system_message = SystemMessage(
        content=f"""You are a compassionate mental health bot and caregiver. Review information about the user and their prior conversation below and respond accordingly.
        Keep responses empathetic, concise and supportive. And remember, always prioritize the user's well-being and mental health.
        {memory.context}"""
    )

    messages = [system_message] + state["messages"]
    response = await llm.ainvoke(messages)
    if not response.tool_calls:
        # Add the new chat turn to the Zep graph
        messages_to_save = []
        for message in messages:
            if isinstance(message, HumanMessage):
                messages_to_save.append(Message(  role="user", name=state["user_name"], content=message.content,),)

        messages_to_save.append(Message(role="assistant", content=response.content))

        zep.thread.add_messages( thread_id=state["thread_id"],  messages=messages_to_save,)

        # Truncate the chat history to keep the state from growing unbounded
        # In this example, we going to keep the state small for demonstration purposes
        # We'll use Zep's Facts to maintain conversation context
        state["messages"] = trim_messages( state["messages"], strategy="last",  token_counter=len,
            max_tokens=3, start_on="human", end_on=("human", "tool"),   include_system=True,)
    else:
        for tool_call in response.tool_calls:
            tool_call["args"]["state"]["first_name"] = state["first_name"]
            tool_call["args"]["state"]["last_name"] = state["last_name"]
            tool_call["args"]["state"]["user_name"] = state["user_name"]
            tool_call["args"]["state"]["thread_id"] = state["thread_id"]

    return {"messages": [response]}

# Define the function that determines whether to continue or not
async def should_continue(state, config):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    # print("tool calls",last_message.tool_calls)
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"




graph_builder = StateGraph(State)

memory = MemorySaver()

graph_builder.add_node("agent", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile(checkpointer=memory)
# print(graph.get_graph().draw_ascii())

def get_user_thread():
    first_name = "Harry"
    last_name = "Liu"
    user_name = first_name + uuid.uuid4().hex[:4]
    thread_id = uuid.uuid4().hex

    zep.user.add(user_id=user_name, first_name=first_name, last_name=last_name)
    zep.thread.create(thread_id=thread_id, user_id=user_name)
    print(first_name, last_name, user_name, thread_id)
    return first_name, last_name, user_name, thread_id

def extract_messages(result):
    output = ""
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            name = "assistant"
        else:
            name = result["user_name"]
        output += f"{name}: {message.content}\n"
    return output.strip()

async def graph_invoke(message: str, first_name: str, last_name: str, user_name:str, thread_id: str, ai_response_only: bool = True,):

    r = await graph.ainvoke(
        {
            "messages": [  {  "role": "user", "content": message, }  ],
            "first_name": first_name,
            "last_name": last_name,
            "user_name":user_name,
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    if ai_response_only:
        return r["messages"][-1].content
    else:
        return extract_messages(r)

def chatbot():
    # first_name, last_name, user_name, thread_id = get_user_thread()
    first_name, last_name, user_name, thread_id = "Harry", "Liu", "Harry2dfc", "25f570725f6a4233ad8942d9d1c6cc79"
    while True:
        try:
            user_input = input("🧑 User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            # print("user input", user_input)
            response = asyncio.run(graph_invoke(user_input, first_name, last_name, user_name, thread_id, ))
            print(f"🤖 Assistant: {response}")
        except Exception as e:

            print("发生错误:")
            traceback.print_exc()
            break
if __name__ == '__main__':
    chatbot()
    # state = State()
    # state["user_name"] = "Harry2dfc"
    # result = search_facts(state, "what were we talking before")
    # print(result)

