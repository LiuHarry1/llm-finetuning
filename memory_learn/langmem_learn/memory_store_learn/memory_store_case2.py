import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

load_dotenv()
llm = ChatTongyi(model="qwen-max", api_key=os.getenv("TONGYI_API_KEY"))

store.put(
    ("users",),
    "user_123",
    {
        "name": "John Smith",
        "language": "English",
    }
)

def get_user_info(config: RunnableConfig) -> str:
    """Look up user info."""
    # Same as that provided to `create_react_agent`
    store = get_store()
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

agent = create_react_agent(
    model=llm,
    tools=[get_user_info],
    store=store
)


# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    config={"configurable": {"user_id": "user_123"}}
)

print(result)
for message in result["messages"]:
    message.pretty_print()


print(store.list_namespaces())