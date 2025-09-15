from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    retriever_weights=[0.6, 0.4],
    similarity_top_k=10,
    num_queries=1,  # set this to 1 to disable query generation
    mode="dist_based_score",
    use_async=True,
    verbose=True,
)

nodes_with_scores = retriever.retrieve(
    "What happened at Interleafe and Viaweb?"
)

for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text[:100]}...\n-----")