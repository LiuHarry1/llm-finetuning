import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import ListMemory, MemoryContent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import autogen_learn.llm_client as llm_client

async def main() -> None:
    # Create a model client.
    model_client = llm_client.model_client

    # Create a list-based memory with some initial content.
    memory = ListMemory()
    await memory.add(MemoryContent(content="User likes pizza.", mime_type="text/plain"))
    await memory.add(MemoryContent(content="User dislikes cheese.", mime_type="text/plain"))

    # Create an AssistantAgent instance with the model client and memory.
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        memory=[memory],
        system_message="You are a helpful assistant.",
    )


    result = await agent.run(task="What is a good dinner idea?")
    print(result.messages[-1].content)  # type: ignore

    print(memory)

    print(memory.content)


asyncio.run(main())
