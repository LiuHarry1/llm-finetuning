#https://github.com/letta-ai/letta

from letta_client import Letta
from letta import llm_api, embeddings


# connect to a local server
client = Letta(base_url="http://localhost:8283")

print(client)

# llm_configs = client.models.list()
# print(f"Available LLM configs: {llm_configs}")
embedding_configs = client.models
print(f"Available embedding configs: {embedding_configs}")

# connect to Letta Cloud
# client = Letta(
#     token="LETTA_API_KEY",
#     project="default-project",
# )

