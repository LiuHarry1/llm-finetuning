from langchain_community.chat_models import ChatTongyi
llm = ChatTongyi( model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

from openai import OpenAI



# -----------------------------
# 初始化 Qwen 客户端
# -----------------------------
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)