from rank_bm25 import BM25Okapi

# 示例数据集
documents = [
    "BM25 is a ranking function used by search engines.",
    "It is based on probabilistic retrieval frameworks.",
    "This method improves document scoring using term frequency and inverse document frequency.",
    "The BM25 algorithm is widely used in information retrieval systems.",
    "bm25 ranking algorithm"
]

# 分词
tokenized_docs = [doc.split() for doc in documents]
# 初始化 BM25 模型
bm25 = BM25Okapi(tokenized_docs)

# 查询
query = "bm25 ranking algorithm"
tokenized_query = query.split()

# 计算相关性得分
scores = bm25.get_scores(tokenized_query)
print(scores)