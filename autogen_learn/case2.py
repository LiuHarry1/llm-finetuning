import asyncio
import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from dotenv import load_dotenv

load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
print(TONGYI_API_KEY)

model_client = OpenAIChatCompletionClient(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-f256c03643e9491fb1ebc278dd958c2d",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "qwen",
        "structured_output": True,
    },
)



assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful assistant."
)


user_proxy = UserProxyAgent("user_proxy", code_execution_config= {"work_dir":"coding"})

