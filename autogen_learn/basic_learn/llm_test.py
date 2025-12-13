from autogen_core.models import UserMessage
from autogen_learn.llm_client import model_client



result = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
print(result)
await model_client.close()
