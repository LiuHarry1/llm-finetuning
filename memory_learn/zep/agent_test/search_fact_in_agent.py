import os

from zep_cloud.client import AsyncZep

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

zep = AsyncZep(api_key=os.environ.get('ZEP_API_KEY'))


@tool
async def search_facts(state: MessagesState, query: str, limit: int = 5):
    """Search for facts in all conversations had with a user.

    Args:
        state (MessagesState): The Agent's state.
        query (str): The search query.
        limit (int): The number of results to return. Defaults to 5.
    Returns:
        list: A list of facts that match the search query.
    """
    search_results = await zep.graph.search(
        user_id=state['user_name'],
        query=query,
        limit=limit,
    )

    return [edge.fact for edge in search_results.edges]


tools = [search_facts]
tool_node = ToolNode(tools)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0).bind_tools(tools)
