from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType

from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.transforms import default_transforms, apply_transforms

kg = KnowledgeGraph()
print(kg)

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

for doc in docs:
    print(doc)
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )
# print("======")
# print(kg)
# print("======")
# for node in     kg.nodes:
#     print(node)



dashScopeEmbeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key = "sk-f256c03643e9491fb1ebc278dd958c2d")

llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = LangchainEmbeddingsWrapper(dashScopeEmbeddings)

trans = default_transforms(documents=docs, llm=generator_llm, embedding_model=generator_embeddings)
apply_transforms(kg, trans)


kg.save("knowledge_graph.json")

# loaded_kg = KnowledgeGraph.load("knowledge_graph.json")
# print(loaded_kg)
#
# from ragas.testset import TestsetGenerator
#
# generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=loaded_kg)
# from ragas.testset.synthesizers import default_query_distribution
#
# query_distribution = default_query_distribution(generator_llm)
# print(query_distribution)
#
# testset = generator.generate(testset_size=10, query_distribution=query_distribution)
# result = testset.to_pandas()
# print(result)
