import os

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
print(TONGYI_API_KEY)

model_client = OpenAIChatCompletionClient(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=TONGYI_API_KEY,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "qwen",
        "structured_output": True,
    },
)