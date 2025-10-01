import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 或 "true"

# 1. 读取数据
import pandas as pd

file_path = "/Users/harry/PycharmProjects/llm-funetuning/retrieve/milvus_learn/quora_duplicate_questions.tsv"
df = pd.read_csv(file_path, sep="\t")
questions = set()
for _, row in df.iterrows():
    obj = row.to_dict()
    questions.add(obj["question1"][:512])
    questions.add(obj["question2"][:512])
    if len(questions) > 500:
        break
docs = list(questions)

print(docs[0])

# 2. Dense embedding: MiniLM
from pathlib import Path


from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models

# 把HF transformers模型包装成SentenceTransformer
word_embedding_model = models.Transformer("/Users/harry/Documents/apps/ml/all-MiniLM-L6-v2")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

dense_embeddings = dense_model.encode(docs, convert_to_numpy=True)


dense_dim = dense_embeddings.shape[1]

# 3. Sparse embedding: BM25
# tokenizing for BM25
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

# 用 BM25 的稀疏向量存到 Milvus，需要把每个 doc 转换为 {token:score} 的 sparse vector
sparse_vectors = []
for doc_tokens in tokenized_docs:
    scores = bm25.get_scores(doc_tokens)  # 这是针对整个语料的分数
    # 简单做法：每个 doc 自身的 token 用1，其他用0（BM25本身是查询时计算分数的）
    # Milvus支持稀疏向量用字典传入 {int:float}
    vec_dict = {}
    for idx, token in enumerate(doc_tokens):
        vec_dict[idx] = 1.0  # 这里只是占位, 可替换成你的BM25权重
    sparse_vectors.append(vec_dict)

# 4. 建立 Milvus collection
connections.connect(uri="./milvus.db")

fields = [
    FieldSchema(
        name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    ),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema = CollectionSchema(fields)

col_name = "hybrid_demo"
if utility.has_collection(col_name):
    Collection(col_name).drop()
col = Collection(col_name, schema, consistency_level="Bounded")

sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)
col.load()

# 5. 插入数据
for i in range(0, len(docs), 50):
    batched_entities = [
        docs[i:i + 50],
        sparse_vectors[i:i + 50],
        dense_embeddings[i:i + 50],
    ]
    col.insert(batched_entities)

print("Number of entities inserted:", col.num_entities)

# 6. 查询
query = input("Enter your search query: ")
print(query)

query_dense_embedding = dense_model.encode([query], convert_to_numpy=True)[0]

# 对稀疏向量我们用BM25重新计算
query_tokens = query.split()
# BM25在查询时算分数，直接用Milvus的稀疏索引需要{int:float}
query_sparse_vec = {}
for idx, token in enumerate(query_tokens):
    query_sparse_vec[idx] = 1.0  # 你也可以用bm25.get_scores(query_tokens)

def dense_search(col, query_dense_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def sparse_search(col, query_sparse_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]

def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]

dense_results = dense_search(col, query_dense_embedding)
sparse_results = sparse_search(col, query_sparse_vec)
hybrid_results = hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_vec,
    sparse_weight=0.7,
    dense_weight=1.0,
)

print("Dense Results:", dense_results)
print("Sparse Results:", sparse_results)
print("Hybrid Results:", hybrid_results)
