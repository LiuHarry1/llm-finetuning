# docker run --name neo4j  -p7474:7474 -p7687:7687 -d  -e NEO4J_AUTH=neo4j/myhome1234  neo4j:latest
from langchain.chains import KnowledgeGraphIndex
from langchain_openai import OpenAI

from langchain.graphs import Neo4jGraph

graph = Neo4jGraph(
    url="bolt://localhost:7687",   # 或者 Aura 提供的 bolt+s://xxxx.databases.neo4j.io
    username="neo4j",
    password="myhome1234"
)


# 初始化 LLM
llm = OpenAI(temperature=0)

# 示例文本
text = """
Harry works at Google.
Harry's friend is Xiaoming.
Xiaoming lives in Beijing.
"""

# 创建知识图谱索引
kg_index = KnowledgeGraphIndex.from_documents(
    documents=[text],
    llm=llm,
    graph=graph
)



