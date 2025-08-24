import asyncio
import traceback
import uuid
from typing import Annotated

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_community.chat_models import ChatTongyi
import os
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from zep_cloud import AsyncZep, Message

class State(TypedDict):
    messages: Annotated[list, add_messages]
    first_name: str
    last_name: str
    thread_id: str
    user_name: str


@tool
async def search_facts(state: State, query: str, limit: int = 5) -> list[str]:
    """Search for facts in all conversations had with a user.

    Args:
        state (State): The Agent's state.
        query (str): The search query.
        limit (int): The number of results to return. Defaults to 5.

    Returns:
        list: A list of facts that match the search query.
    """
    print("starting to call search facts method", state)
    edges = await zep.graph.search(
        user_id=state["user_name"], text=query, limit=limit, search_scope="edges"
    )
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
    nodes = await zep.graph.search(
        user_id=state["user_name"], text=query, limit=limit, search_scope="nodes"
    )
    return [node.summary for node in nodes]



llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")


load_dotenv(override=True)
ZEP_API_KEY = os.getenv("ZEP_API_KEY")
zep = AsyncZep(api_key=ZEP_API_KEY)

tools = [search_facts, search_nodes]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

graph_builder = StateGraph(State)

async def chatbot(state: State):
    memory = await zep.thread.get_user_context(state["thread_id"])

    system_message = SystemMessage(
        content=f"""You are a compassionate mental health bot and caregiver. Review information about the user and their prior conversation below and respond accordingly.
        Keep responses empathetic and supportive. And remember, always prioritize the user's well-being and mental health.

        {memory.context}"""
    )

    messages = [system_message] + state["messages"]

    response = await llm.ainvoke(messages)

    # Add the new chat turn to the Zep graph
    messages_to_save = [
        Message(
            role="user",
            name=state["first_name"] + " " + state["last_name"],
            content=state["messages"][-1].content,
        ),
        Message(role="assistant", content=response.content),
    ]

    await zep.thread.add_messages(
        thread_id=state["thread_id"],
        messages=messages_to_save,
    )

    # Truncate the chat history to keep the state from growing unbounded
    # In this example, we going to keep the state small for demonstration purposes
    # We'll use Zep's Facts to maintain conversation context
    state["messages"] = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,
        max_tokens=3,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    print(f"Messages in state: {state['messages']}")


    return {"messages": [response]}

graph_builder = StateGraph(State)

memory = MemorySaver()


# Define the function that determines whether to continue or not
async def should_continue(state, config):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


graph_builder.add_node("agent", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})

graph_builder.add_edge("tools", "agent")


graph = graph_builder.compile(checkpointer=memory)
print(graph.get_graph().draw_ascii())

async def get_user_thread():
    # first_name = "Harry"
    # last_name = "Liu"
    # user_name = first_name + uuid.uuid4().hex[:4]
    # thread_id = uuid.uuid4().hex
    #
    # await zep.user.add(user_id=user_name, first_name=first_name, last_name=last_name)
    # await zep.thread.create(thread_id=thread_id, user_id=user_name)
    # print(first_name, last_name, user_name, thread_id)
    # return first_name, last_name, user_name, thread_id
    return "Harry","Liu", "Harry7494", "93f3277e2f6047db9bc17df1bbc853e1"


async def view_user_thread(thread_id):
    memory = await zep.thread.get_user_context(thread_id=thread_id)
    print(memory.context)


async def main():
    first_name, last_name, user_name, thread_id = await get_user_thread()


    def extract_messages(result):
        output = ""
        for message in result["messages"]:
            if isinstance(message, AIMessage):
                name = "assistant"
            else:
                name = result["user_name"]
            output += f"{name}: {message.content}\n"
        return output.strip()


    async def graph_invoke(
        message: str,
        first_name: str,
        last_name: str,
        thread_id: str,
        ai_response_only: bool = True,
    ):
        r = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
                "first_name": first_name,
                "last_name": last_name,
                "thread_id": thread_id,
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        if ai_response_only:
            return r["messages"][-1].content
        else:
            return extract_messages(r)


    # r = await graph_invoke(
    #     "Hi there?",
    #     first_name,
    #     last_name,
    #     thread_id,
    # )
    #
    # print(r)

    # r = await graph_invoke(
    #     """
    #     I'm fine. But have been a bit stressful lately. Mostly work related.
    #     But also my dog. I'm worried about her.
    #     """,
    #     first_name,
    #     last_name,
    #     thread_id,
    # )

    # r = await graph_invoke(
    #     "She ate my shoes which were expensive.",
    #     first_name,
    #     last_name,
    #     thread_id,
    # )

    # r = await graph_invoke(
    #     "What are we talking about?",
    #     first_name,
    #     last_name,
    #     thread_id,
    # )

    r = await graph_invoke(
        "What have I said about my job?",
        first_name,
        last_name,
        thread_id,
    )


    print(r)


if __name__ == '__main__':
    asyncio.run(main())
    # asyncio.run(view_user_thread("93f3277e2f6047db9bc17df1bbc853e1"))
