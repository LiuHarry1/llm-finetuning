import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langmem import create_memory_manager
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatTongyi(model="qwen-max", api_key=os.getenv("TONGYI_API_KEY"))

class Episode(BaseModel):
    """Write the episode from the perspective of the agent within it. Use the benefit of hindsight to record the memory, saving the agent's key internal thought process so it can learn over time."""

    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description="Internal reasoning process and observations of the agent in the episode that let it arrive"
        ' at the correct action and result. "I ..."',
    )
    action: str = Field(
        ...,
        description="What was done, how, and in what format. (Include whatever is salient to the success of the action). I ..",
    )
    result: str = Field(
        ...,
        description="Outcome and retrospective. What did you do well? What could you do better next time? I ...",
    )

manager = create_memory_manager(
    llm,
    schemas=[Episode],
    instructions="Extract examples of successful explanations, capturing the full chain of reasoning. Be concise in your explanations and precise in the logic of your reasoning.",
    enable_inserts=True,
)

conversation = [
    {
        "role": "user",
        "content": "What's a binary tree? I work with family trees if that helps",
    },
    {
        "role": "assistant",
        "content": "A binary tree is like a family tree, but each parent has at most 2 children. Here's a simple example:\n   Bob\n  /  \\\nAmy  Carl\n\nJust like in family trees, we call Bob the 'parent' and Amy and Carl the 'children'.",
    },
    {
        "role": "user",
        "content": "Oh that makes sense! So in a binary search tree, would it be like organizing a family by age?",
    },
]

episodes = manager.invoke({"messages": conversation})
print(episodes[0])

# ExtractedMemory(
#     id="2e5c551f-58a7-40c2-96b3-cabdfa5ccb31",
#     content=Episode(
#         observation="In a teaching interaction, I used a family tree analogy to explain binary trees, which led to a successful understanding. The student then made an insightful connection to binary search trees and age ordering.",
#         thoughts="I noticed that connecting abstract data structures to familiar concepts like family relationships made the concept more accessible. The student's quick grasp and ability to extend the analogy to binary search trees showed the effectiveness of this approach. Using relatable examples helps bridge the gap between technical concepts and everyday understanding.",
#         action='I explained binary trees using a family tree metaphor, drawing a simple diagram with "Bob" as parent and "Amy" and "Carl" as children. This visualization provided a concrete, relatable example that built on the student\'s existing knowledge of family trees.',
#         result="The explanation was highly successful, evidenced by the student's immediate comprehension (\"Oh that makes sense!\") and their ability to make the cognitive leap to understanding binary search trees' ordering property. For future explanations, I should continue using familiar analogies while being prepared to build upon them for more complex concepts. The family tree analogy proved particularly effective for explaining hierarchical structures.",
#     ),
# )