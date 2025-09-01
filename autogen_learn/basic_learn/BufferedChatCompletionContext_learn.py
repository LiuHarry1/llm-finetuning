import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext
import autogen_learn.llm_client as llm_client


async def main() -> None:
    # Create a model client.
    model_client = llm_client.model_client

    # Create a model context that only keeps the last 2 messages (1 user + 1 assistant).
    model_context = BufferedChatCompletionContext(buffer_size=3)

    # Create an AssistantAgent instance with the model client and context.
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        model_context=model_context,
        system_message="You are a helpful assistant.",
    )

    result = await agent.run(task="Name two cities in North America.")
    print(result.messages[-1].content)  # type: ignore

    result = await agent.run(task="My favorite color is blue.")
    print(result.messages[-1].content)  # type: ignore

    result = await agent.run(task="Did I ask you any question?")
    print(result.messages[-1].content)  # type: ignore

    print("======")
    messages = await model_context.get_messages();
    print(messages)
    for message in messages:
        print(message)


asyncio.run(main())
