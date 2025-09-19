import os

from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate

load_dotenv()
# 初始化 LLM（这里用 OpenAI，如果你用别的模型，可以替换）

llm = ChatTongyi(model="qwen-plus",temperature=0, api_key=os.getenv("DASHSCOPE_API_KEY") )

# 模拟加载文档（实际可从文件、数据库等获取）
docs = [
    Document(page_content="人工智能正在快速发展，对社会的各个方面都产生了深远的影响。"),
    Document(page_content="尤其是在教育、医疗、金融等领域，AI 技术的应用不断扩展。"),
    Document(page_content="然而，AI 的快速普及也带来了伦理和隐私方面的挑战。")
]


# 选择总结链的模式："stuff" | "map_reduce" | "refine"
chain = load_summarize_chain(llm,
                             chain_type="refine",
                             verbose=True
                        )

# 运行总结
summary = chain.run(docs)

print("总结结果：")
print(summary)
