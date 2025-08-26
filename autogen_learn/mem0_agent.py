import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.mem0 import Mem0Memory
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()
MEM0_API_KEY = os.getenv("MEM0_API_KEY")

TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
print(TONGYI_API_KEY)

model_client = OpenAIChatCompletionClient(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=TONGYI_API_KEY,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "qwen",
        "structured_output": True,
        "multiple_system_messages": True,  # <-- add this
    },
)

# Initialize Mem0 cloud memory (requires API key)
# For local deployment, use is_cloud=False with appropriate config
mem0_memory = Mem0Memory(
    is_cloud=True,
    limit=5,  # Maximum number of memories to retrieve
    api_key=MEM0_API_KEY
)

async def main():

    # Add user preferences to memory
    await mem0_memory.add(
        MemoryContent(
            content="The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "units"},
        )
    )

    await mem0_memory.add(
        MemoryContent(
            content="Meal recipe must be vegan",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "dietary"},
        )
    )

    async def get_weather(city: str, units: str = "imperial") -> str:
        if units == "imperial":
            return f"The weather in {city} is 73 °F and Sunny."
        elif units == "metric":
            return f"The weather in {city} is 23 °C and Sunny."
        else:
            return f"Sorry, I don't know the weather in {city}."


    # Create assistant with mem0 memory
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        tools=[get_weather],
        memory=[mem0_memory],
    )

    # Ask about the weather
    stream = assistant_agent.run_stream(task="What are my dietary preferences?")
    await Console(stream)
asyncio.run(main())
