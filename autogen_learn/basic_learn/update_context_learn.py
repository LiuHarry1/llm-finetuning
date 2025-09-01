import asyncio
from autogen_core.memory import ListMemory, MemoryContent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import UserMessage, AssistantMessage


async def main() -> None:
    # Initialize memory
    memory = ListMemory(name="chat_history")

    # Add memory content
    content = MemoryContent(content="User prefers formal language", mime_type="text/plain")
    await memory.add(content)
    await memory.add(MemoryContent(content="New preference", mime_type="text/plain"))

    # Directly modify memory contents
    # memory.content = [MemoryContent(content="New preference", mime_type="text/plain")]

    # Create a model context
    model_context = BufferedChatCompletionContext(buffer_size=5)
    await model_context.add_message(UserMessage(content = "test1", source = "user" ))
    await model_context.add_message(AssistantMessage(content ="assistant test1", source="assistant"))
    await model_context.add_message(UserMessage(content="test2", source="user"))

    # Update a model context with memory
    await memory.update_context(model_context)

    # See the updated model context
    print(await model_context.get_messages())

    messages = await model_context.get_messages()
    for message in messages:
        print(message)


asyncio.run(main())