import os

from langchain_community.retrievers import ZepCloudRetriever
from zep_cloud.client import AsyncZep
from zep_cloud import Message, ThreadGetUserContextRequestMode
import asyncio
#https://app.getzep.com/projects/ff41087d-e0f6-496d-8cea-e2e05be96d20/playground

zep_api_key = "z_1dWlkIjoiY2EzYzc1ZjMtYmM4My00N2I3LWFkZGYtNzBiM2NjMzM5NmRmIn0.3j67lAIWi8r5zVs9JChzXUHkGlYKZg74kEs_pUwUxhmDduj76IPLnDiVzG3MaapDGm8hdDLeH0GTxDuglH2A_w"

def test_zep1():
    zep = AsyncZep(api_key="z_1dWlkIjoiY2EzYzc1ZjMtYmM4My00N2I3LWFkZGYtNzBiM2NjMzM5NmRmIn0.3j67lAIWi8r5zVs9JChzXUHkGlYKZg74kEs_pUwUxhmDduj76IPLnDiVzG3MaapDGm8hdDLeH0GTxDuglH2A_w")

    # print("Current Context:", memory.context)


async def test_zep():
    # 初始化 Zep 客户端

    client = AsyncZep(api_key=zep_api_key)

    user_id = "user_123"
    thread_id = "thread_123"

    # 添加/更新用户信息
    # await client.user.add(user_id=user_id, first_name="Harry")

    # 创建线程
    # await client.thread.create(thread_id=thread_id, user_id=user_id)

    # 添加消息
    # messages = [
    #     Message(name="Harry", role="user", content="Hi, I'm looking for a sci-fi movie recommendation."),
    #     Message(name="Assistant", role="assistant", content="How about 'Dune'? It's a great sci-fi movie.")
    # ]
    # await client.thread.add_messages(thread_id=thread_id, messages=messages)
    #
    # context = await client.thread.get_user_context(thread_id=thread_id)
    # print("Context:", context)

    results = await client.graph.search(user_id=user_id, query="Dune", limit=3)
    print("Graph Facts:", [edge.fact for edge in results.edges])

    results = await client.thread.search(
        thread_id=thread_id,
        query="sci-fi movie",
        top_k=2
    )
    for m in results.messages:
        print(m.role, ":", m.content)

async def search_in_history():

    user_id = "user_123"
    thread_id = "thread_123"

    # 初始化 Zep Cloud 客户端
    client = AsyncZep(api_key=zep_api_key)


    # 获取线程历史
    context = await client.thread.get_user_context(thread_id=thread_id, mode="basic")

    print("Context:", context)

    context = await client.thread.get_user_context(thread_id=thread_id, mode="summary")

    print("Context:", context)



"""

FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges
# format: FACT (Date range: from - to)
<FACTS>
  - How about 'Dune'? It's a great sci-fi movie. (2025-08-21 02:44:31 - present)
  - Harry is looking for a sci-fi movie recommendation. (2025-08-21 02:59:55 - present)
  - Hi, I'm looking for a sci-fi movie recommendation. (2025-08-21 02:44:31 - present)
  - Dune is a sci-fi movie. (2025-08-21 02:58:26 - present)
  - It's a great sci-fi movie. (2025-08-21 02:59:55 - present)
</FACTS>

# These are the most relevant entities
# Name: ENTITY_NAME
# Label: entity_label (if present)
# Attributes: (if present)
#   attr_name: attr_value
# Summary: entity summary
<ENTITIES>
  - Name: sci-fi movie
    Label: Topic
    Attributes:
      domain: sci-fi
      expertise_level: movie
    Summary: The user is looking for a sci-fi movie recommendation. The assistant suggested 'Dune'.
  - Name: Harry
    Label: User
    Attributes:
      email: 
      first_name: Harry
      last_name: 
      role_type: user
      user_id: user_123
    Summary: user with the id of user_123, name of Harry. Harry is looking for a sci-fi movie recommendation.
</ENTITIES>

"""

"""
• Harry requested a sci-fi movie recommendation on 2025-08-21 02:44:31 (2025-08-21 02:44:31 - present)
• The assistant recommended 'Dune' as a sci-fi movie on 2025-08-21 02:44:31 (2025-08-21 02:44:31 - present)
• 'Dune' was described as a great sci-fi movie on 2025-08-21 02:59:55 (2025-08-21 02:59:55 - present)"

"""

if __name__ == '__main__':
    asyncio.run(test_zep())
    # asyncio.run(search_in_history())
