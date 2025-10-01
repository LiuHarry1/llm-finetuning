# ------------------------------
# 3. Sparse embedding: BM25 + Milvus稀疏向量
# ------------------------------
import pandas as pd

file_path = "quora_duplicate_questions.tsv"
df = pd.read_csv(file_path, sep="\t")
questions = set()
for _, row in df.iterrows():
    obj = row.to_dict()
    questions.add(obj["question1"][:512])
    questions.add(obj["question2"][:512])
    if len(questions) > 500:  # Skip this if you want to use the full dataset
        break

docs = list(questions)

print(docs[0])


from rank_bm25 import BM25Okapi

# 1. Tokenize 文档
tokenized_docs = [doc.split() for doc in docs]

# 2. BM25 模型
bm25 = BM25Okapi(tokenized_docs)

# 3. 构建全局词表: token -> unique_id
vocab = {}
next_id = 0
for doc_tokens in tokenized_docs:
    for token in doc_tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1

# 4. 构造每条文档的稀疏向量 {token_id:1.0}
sparse_vectors = []
for doc_tokens in tokenized_docs:
    vec_dict = {}
    for token in doc_tokens:
        token_id = vocab[token]
        vec_dict[token_id] = 1.0  # 可以换成 TF 或 TF-IDF 权重
    sparse_vectors.append(vec_dict)

# 5. 构造查询的稀疏向量
def get_query_sparse_vector(query: str, vocab: dict):
    query_tokens = query.split()
    vec_dict = {}
    for token in query_tokens:
        if token in vocab:
            vec_dict[vocab[token]] = 1.0  # 可以用 BM25.get_scores([query_tokens]) 进一步加权
    return vec_dict

query = "what is machine learning"

# 用法：
query_sparse_vec = get_query_sparse_vector(query, vocab)

sparse_results = sparse_search(col, query_sparse_vec)