import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi, ChatOllama
from langchain_community.embeddings import DashScopeEmbeddings, OllamaEmbeddings
from langgraph.func import entrypoint
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from langmem import create_manage_memory_tool, create_search_memory_tool

load_dotenv()

embeddings = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=os.getenv("TONGYI_API_KEY"))

# embeddings = OllamaEmbeddings(model="llama3.1:8b")

# Set up store and checkpointer
store = InMemoryStore(index={"dims": 1536, "embed": embeddings,})

my_llm = ChatTongyi(model="qwen-plus", api_key=os.getenv("TONGYI_API_KEY"))
# my_llm = ChatOllama(model="llama3.1:8b", temperature=0.8)  # 确认名字和 ollama list 一致



# my_llm = init_chat_model(
#     model="llama3.1:8b",          # 模型名直接写 ollama list 里看到的
#     model_provider="ollama",      # 手动指定 provider
#     model_kwargs={"temperature": 0}
# )


def prompt(state):
    """Prepare messages with context from existing memories."""
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a memory manager. Extract and manage all important knowledge, rules, and events using the provided tools.



Existing memories:
<memories>
{memories}
</memories>

Use the manage_memory tool to update and contextualize existing memories, create new ones, or delete old ones that are no longer valid.
You can also expand your search of existing memories to augment using the search tool."""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


# Create the memory extraction agent
manager = create_react_agent(
    my_llm,
    prompt=prompt,
    tools=[
        # Agent can create/update/delete memories
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
)


# Run extraction in background
@entrypoint(store=store)
def app(messages: list):
    response = my_llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            *messages,
        ]
    )

    # Extract and store triples (Uses store from @entrypoint context)
    manager.invoke({"messages": messages})
    return response


app.invoke(
    [
        {
            "role": "user",
            "content": "Alice manages the ML team and mentors Bob, who is also on the team.",
        }
    ]
)

print(store.search(("memories",)))

