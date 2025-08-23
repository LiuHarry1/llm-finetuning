from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# 文档列表
docs = [Document(page_content="文本1"), Document(page_content="文本2")]

# 加载 map-reduce 摘要链
chain = load_summarize_chain(llm, chain_type="map_reduce")

# 执行摘要
summary = chain.run(docs)
