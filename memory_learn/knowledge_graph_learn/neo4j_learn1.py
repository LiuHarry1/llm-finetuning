# docker run --name neo4j  -p7474:7474 -p7687:7687 -d  -e NEO4J_AUTH=neo4j/myhome1234  neo4j:latest
#https://python.langchain.com/docs/how_to/graph_constructing/


# docker run --name neo4j -p7474:7474 -p7687:7687 -d -e NEO4J_AUTH=neo4j/myhome1234  -e NEO4JLABS_PLUGINS='["apoc"]'  -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true  -e NEO4J_apoc_import_file_use__neo4j__config=true neo4j:latest


import asyncio
import os

from langchain_neo4j import Neo4jGraph
from langchain_community.chat_models import ChatTongyi

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "myhome1234"

async def test():
    llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d" )


    graph = Neo4jGraph(refresh_schema=False)

    from langchain_experimental.graph_transformers import LLMGraphTransformer


    llm_transformer = LLMGraphTransformer(llm=llm)

    from langchain_core.documents import Document

    text = """
    Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    She was, in 1906, the first woman to become a professor at the University of Paris.
    """
    documents = [Document(page_content=text)]
    graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")

    graph.add_graph_documents(graph_documents)

if __name__ == '__main__':
    asyncio.run(test())



