import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

load_dotenv()
# 配置 API Key
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

# 初始化 ChatTongyi，开启流式
llm = ChatTongyi(
    model="qwen-plus",
    temperature=0.7,
    api_key=api_key,
    streaming=True
)

# 测试输入
messages = [HumanMessage(content="请用Python写一个带注释的二分查找函数")]

print("=== 流式输出开始 ===\n")
full_text = ""

# 调用 .stream() 方法
for chunk in llm.stream(messages):
    # chunk 是 ChatGenerationChunk
    token = chunk.text()   # ✅ 注意这里要加 () 调用方法
    if token:  # 有时候 chunk 里可能为空
        full_text += token
        print(token, end="", flush=True)

print("\n\n=== 完整内容 ===")
print(full_text)
