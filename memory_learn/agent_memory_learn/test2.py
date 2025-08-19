from langchain_community.chat_models import ChatTongyi

llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-f256c03643e9491fb1ebc278dd958c2d"
)

resp = llm.invoke("你好，帮我总结一下LangGraph的用途")
print(resp.content)