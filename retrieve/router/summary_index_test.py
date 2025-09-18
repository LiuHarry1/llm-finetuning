from llama_index.core import SummaryIndex, Document

# 准备一些文档
docs = [
    Document(text="LlamaIndex 是一个用于构建 LLM 应用的框架。"),
    Document(text="它支持多种索引，比如 VectorIndex, SummaryIndex 等。"),
    Document(text="RouterRetriever 可以结合不同的检索器。"),
]

# 建立 SummaryIndex
index = SummaryIndex.from_documents(docs)

# 获取 retriever
retriever = index.as_retriever()

# 查询
query = "什么是 LlamaIndex？"
results = retriever.retrieve(query)

for r in results:
    print(r)
