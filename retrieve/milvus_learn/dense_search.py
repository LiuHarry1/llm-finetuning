import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, models
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

# ------------------------------
# 1. 读取数据
# ------------------------------
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
print("Sample doc:", docs[0])

# ------------------------------
# 2. Dense embedding: MiniLM
# ------------------------------
word_embedding_model = models.Transformer("/Users/harry/Documents/apps/ml/all-MiniLM-L6-v2")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

dense_embeddings = dense_model.encode(docs, convert_to_numpy=True)
dense_dim = dense_embeddings.shape[1]

# ------------------------------
# 3. Sparse embedding: BM25 + 全局词表 + TF
# ------------------------------
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

# 全局词表
vocab = {}
next_id = 0
for doc_tokens in tokenized_docs:
    for token in doc_tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

# 文档稀疏向量
sparse_vectors = []
for doc_tokens in tokenized_docs:
    vec_dict = {}
    for token in doc_tokens:
        token_id = vocab[token]
        vec_dict[token_id] = vec_dict.get(token_id, 0.0) + 1.0  # TF
        # vec_dict[vocab[token]] = bm25.get_scores(token, doc_tokens)  # 或者自定义计算
    sparse_vectors.append(vec_dict)

# 查询稀疏向量
def get_query_sparse_vector(query: str, vocab: dict):
    query_tokens = query.split()
    vec_dict = {}
    for token in query_tokens:
        if token in vocab:
            vec_dict[vocab[token]] = vec_dict.get(vocab[token], 0.0) + 1.0
            # vec_dict[vocab[token]] = bm25.get_scores(token, doc_tokens)  # 或者自定义计算
    return vec_dict

# ------------------------------
# 4. 建立 Milvus collection
# ------------------------------
connections.connect(uri="./milvus.db")

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema = CollectionSchema(fields)

col_name = "hybrid_demo"
if utility.has_collection(col_name):
    Collection(col_name).drop()
col = Collection(col_name, schema, consistency_level="Bounded")

# 创建索引
col.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
col.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
col.load()

# 插入数据
for i in range(0, len(docs), 50):
    batched_entities = [
        docs[i:i + 50],
        sparse_vectors[i:i + 50],
        dense_embeddings[i:i + 50],
    ]
    col.insert(batched_entities)

print("Number of entities inserted:", col.num_entities)

# ------------------------------
# 5. 搜索函数
# ------------------------------
def dense_search(col, query_dense_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search([query_dense_embedding],
                     anns_field="dense_vector",
                     limit=limit,
                     output_fields=["text"],
                     param=search_params)[0]
    return [hit.get("text") for hit in res]

def sparse_search(col, query_sparse_embedding, limit=10):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search([query_sparse_embedding],
                     anns_field="sparse_vector",
                     limit=limit,
                     output_fields=["text"],
                     param=search_params)[0]
    return [hit.get("text") for hit in res]

def hybrid_search(col, query_dense_embedding, query_sparse_embedding, sparse_weight=1.0, dense_weight=1.0, limit=10):
    dense_req = AnnSearchRequest([query_dense_embedding], "dense_vector", {"metric_type": "IP", "params": {}}, limit=limit)
    sparse_req = AnnSearchRequest([query_sparse_embedding], "sparse_vector", {"metric_type": "IP", "params": {}}, limit=limit)
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search([sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"])[0]
    return [hit.get("text") for hit in res]

# ------------------------------
# 6. 输入查询 & 检索
# ------------------------------
query = input("Enter your search query: ")
query_dense_embedding = dense_model.encode([query], convert_to_numpy=True)[0]
query_sparse_vec = get_query_sparse_vector(query, vocab)

dense_results = dense_search(col, query_dense_embedding)
sparse_results = sparse_search(col, query_sparse_vec)
hybrid_results = hybrid_search(col, query_dense_embedding, query_sparse_vec, sparse_weight=0.7, dense_weight=1.0)

print("\nDense Results:", dense_results)
print("\nSparse Results:", sparse_results)
print("\nHybrid Results:", hybrid_results)
