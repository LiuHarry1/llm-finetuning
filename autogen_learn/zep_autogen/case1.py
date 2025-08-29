import os
import uuid
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.memory import MemoryContent, MemoryMimeType
from dotenv import load_dotenv
from zep_cloud.client import AsyncZep, Zep
from zep_autogen import ZepUserMemory

# Initialize Zep client
load_dotenv(override=True)
ZEP_API_KEY = os.getenv("ZEP_API_KEY")


llm = OpenAIChatCompletionClient(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("TONGYI_API_KEY"),
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "qwen",
        "structured_output": True,
        "multiple_system_messages": True,  # 👈 加上这个
    },
)
zep_client = AsyncZep(api_key=os.environ.get("ZEP_API_KEY"))


async def get_user_thread():
    user_id = f"user_{uuid.uuid4().hex[:16]}"
    thread_id = f"thread_{uuid.uuid4().hex[:16]}"

    # Create user (required before using memory)
    try:
        await zep_client.user.add(
            user_id=user_id,
            email="alice@example.com",
            first_name="Alice"
        )
    except Exception as e:
        print(f"User might already exist: {e}")

    # Create thread (required for conversation memory)
    try:
        await zep_client.thread.create(thread_id=thread_id, user_id=user_id)
    except Exception as e:
        print(f"Thread creation failed: {e}")
    print(user_id, thread_id)
    return user_id, thread_id


async def add_message(memory, message: str, role: str, name: str = None):
    """Store a message in Zep memory following AutoGen standards."""
    metadata = {"type": "message", "role": role}
    if name:
        metadata["name"] = name

    await memory.add(MemoryContent(
        content=message,
        mime_type=MemoryMimeType.TEXT,
        metadata=metadata
    ))


async def chatbot():
    # user_id, thread_id = await get_user_thread()
    user_id , thread_id = "user_922f5e2b25bb40b0", "thread_e3199cd81a324f2b"

    # Create user memory with configuration
    memory = ZepUserMemory(
        client=zep_client,
        user_id=user_id,
        thread_id=thread_id,
        thread_context_mode="summary"
    )

    # Create agent with Zep memory
    agent = AssistantAgent(
        name="MemoryAwareAssistant",
        model_client=llm,
        memory=[memory],
        system_message="You are a helpful assistant with persistent memory."
    )

    # Example conversation with memory persistence
    user_message = "My name is Alice and I love hiking in the mountains."
    print(f"User: {user_message}")

    # Store user message
    await add_message(memory, user_message, "user", "Alice")

    # Run agent - it will automatically retrieve context via update_context()
    response = await agent.run(task=user_message)
    agent_response = response.messages[-1].content
    print(f"Agent: {agent_response}")

    # Store agent response
    await add_message(memory, agent_response, "assistant")


if __name__ == '__main__':
    asyncio.run(chatbot())
