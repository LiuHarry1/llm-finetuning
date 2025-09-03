import asyncio
import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi, ChatOllama
from pydantic import BaseModel
from langmem import create_memory_manager

load_dotenv()
llm = ChatTongyi(model="qwen-max", api_key=os.getenv("TONGYI_API_KEY"))

class SimpleOllama(ChatOllama):
    def bind_tools(self, *args, **kwargs):
        return self

# llm = SimpleOllama(model="llama3.1:8b", temperature=0)

class PreferenceMemory(BaseModel):
    """Store the user's preference"""
    category: str
    preference: str
    context: str

manager = create_memory_manager(
    llm,
    schemas=[PreferenceMemory]
)

async def main():

    # Same conversation, but with structured output
    conversation = [
        {"role": "user", "content": "I prefer dark mode in all my apps"},
        {"role": "assistant", "content": "I'll remember that preference"}
    ]
    memories =  await manager(conversation)
    print(memories)
    # print(memories[0][1])
    # Output:
    # PreferenceMemory(
    #     category="ui",
    #     preference="dark_mode",
    #     context="User explicitly stated preference for dark mode in all applications"
    # )

    conversation = [
        {
            "role": "user",
            "content": "Actually I changed my mind, dark mode hurts my eyes",
        },
        {"role": "assistant", "content": "I'll update your preference"},
    ]

    # The manager will upsert; working with the existing memory instead of always creating a new one
    updated_memories = await manager.ainvoke(
        {"messages": conversation, "existing": memories}
    )
    print(updated_memories)

if __name__ == '__main__':
    asyncio.run(main())