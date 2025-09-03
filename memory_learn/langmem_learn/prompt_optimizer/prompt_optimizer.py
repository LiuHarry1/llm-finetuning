import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import ChatOllama
from langmem import create_prompt_optimizer


load_dotenv()
# llm = ChatTongyi(model="qwen-plus", api_key=os.getenv("TONGYI_API_KEY"),  model_kwargs={"tool_choice": "auto"} )


llm = ChatOllama(
    model="llama3.1:8b",   # ollama 里模型的名字，比如 "llama3" 或 "llama3:8b"
    # model = "deepseek-r1:1.5b",
    temperature=0,    # 可选，保证输出更稳定
)

# Multiple conversations showing what to improve
trajectories = [
    # Conversation with no annotations (just the conversation)
    (
        [
            {"role": "user", "content": "Tell me about Mars"},
            {"role": "assistant", "content": "Mars is the fourth planet..."},
            {"role": "user", "content": "I wanted more about its moons"},
        ],
        None,
    ),
    (
        [
            {"role": "user", "content": "What are Mars' moons?"},
            {
                "role": "assistant",
                "content": "Mars has two moons: Phobos and Deimos...",
            },
        ],
        {
            "score": 0.9,
            "comment": "Should include more details and recommended follow-up questions",
        },
    ),
    # Annotations can be of different types, such as edits/revisions!
    (
        [
            {"role": "user", "content": "Compare Mars and Earth"},
            {"role": "assistant", "content": "Mars and Earth have many differences..."},
        ],
        {"revised": "Earth and Mars have many similarities and differences..."},
    ),
]

optimizer = create_prompt_optimizer(
    llm,
    kind="metaprompt",
    config={"max_reflection_steps": 1, "min_reflection_steps": 0},
)
updated = optimizer.invoke(
    {"trajectories": trajectories, "prompt": "You are a planetary science expert"}
)
print(updated)