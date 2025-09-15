import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import SparseVector

# ===============================
# 1. 加载本地 Transformers 模型
# ===============================
model_name = '/Users/harry/Documents/apps/ml/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def encode(texts):
    """将文本转成句子 embedding (mean pooling)"""
    if isinstance(texts, str):
        texts = [texts]
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return embeddings

# ===============================
# 2. 文档数据
# ===============================
docs = [
    "Cats are wonderful pets.",
    "Dogs are loyal animals.",
    "Birds can fly in the sky.",
    "Fish swim in the water.",
    "Retrieval-augmented generation combines search with language models.",
]

# ===============================
# 3. BM25 稀疏向量
# ===============================
tokenized_docs = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

def bm25_vectorize(doc, bm25):
    """生成 Qdrant SparseVector"""
    tokens = doc.lower().split()
    scores = bm25.get_scores(tokens)
    indices = [i for i, s in enumerate(scores) if s > 0]
    values = [float(scores[i]) for i in indices]
    return SparseVector(indices=indices, values=values)

# ===============================
# 4. Qdrant 集合创建
# ===============================
client = QdrantClient("http://localhost:6333")
collection_name = "bm25_dense_local"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(
            size=384,  # 本地模型输出维度
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=True)
        )
    }
)

# ===============================
# 5. 插入文档
# ===============================
dense_embeddings = encode(docs).tolist()
points = []
for idx, (doc, dense_vec) in enumerate(zip(docs, dense_embeddings), start=1):
    sparse_vec = bm25_vectorize(doc, bm25)
    points.append(
        models.PointStruct(
            id=idx,
            vector={"dense": dense_vec, "sparse": sparse_vec},
            payload={"text": doc},
        )
    )

client.upsert(collection_name=collection_name, points=points)

# ===============================
# 6. 查询
# ===============================
query = "Tell me about animals that are good friends"
# query_dense = encode(query).tolist()
query_dense = encode(query)[0].tolist()
query_sparse_vector = bm25_vectorize(query, bm25)

results = client.query_points(
    collection_name=collection_name,
    prefetch=[
        models.Prefetch(query=query_sparse_vector, using="sparse", limit=5),
        models.Prefetch(query=query_dense, using="dense", limit=5),
    ],
    query=models.FusionQuery(fusion=models.Fusion.DBSF),
    limit=5,
)

# ===============================
# 7. 输出结果
# ===============================
print(f"\nQuery: {query}\nTop results:")
for hit in results.points:
    print(f"Score: {hit.score:.4f} | Text: {hit.payload['text']}")
