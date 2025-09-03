from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langmem import create_prompt_optimizer

load_dotenv()

# 定义一个 wrapper，绕过 bind_tools
class SimpleOllama(ChatOllama):
    def bind_tools(self, *args, **kwargs):
        return self

llm = SimpleOllama(model="llama3.1:8b", temperature=0)

trajectories = [
    (
        [
            {"role": "user", "content": "Tell me about Mars"},
            {"role": "assistant", "content": "Mars is the fourth planet..."},
            {"role": "user", "content": "I wanted more about its moons"},
        ],
        None,
    )
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
