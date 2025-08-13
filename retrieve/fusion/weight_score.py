from llama_index.core.retrievers import HybridRetriever

retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    sparse_retriever=sparse_retriever,
    alpha=0.7,  # 向量占比，sparse 为 1-alpha
)
