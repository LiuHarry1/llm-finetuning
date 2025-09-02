from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

model = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

def tool() -> None:
    """Testing tool."""
    ...

def pre_model_hook() -> None:
    """Pre-model hook."""
    ...

def post_model_hook() -> None:
    """Post-model hook."""
    ...

class ResponseFormat(BaseModel):
    """Response format for the agent."""
    result: str

agent = create_react_agent(
    model,
    tools=[tool],
    pre_model_hook=pre_model_hook,
    post_model_hook=post_model_hook,
    response_format=ResponseFormat,
)

# Visualize the graph
# For Jupyter or GUI environments:
# agent.get_graph().draw_mermaid_png()

# To save PNG to file:
png_data = agent.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

# For terminal/ASCII output:
agent.get_graph().draw_ascii()