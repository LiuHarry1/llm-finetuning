from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# 连接 Qdrant
client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

collection_name = "test1"

# 1. 创建 collection（只需执行一次）
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3, distance=Distance.COSINE)  # 向量维度 = 3
)
print("✅ 已创建 collection")

# 2. 插入向量
client.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(id=1, vector=[0.1, 0.2, 0.3], payload={"text": "vector 1"}),
        PointStruct(id=2, vector=[0.11, 0.19, 0.29], payload={"text": "vector 2"}),
        PointStruct(id=3, vector=[0.9, 0.8, 0.7], payload={"text": "vector 3"}),
    ]
)
print("✅ 已插入数据")

# 3. 搜索
hits = client.search(
    collection_name=collection_name,
    query_vector=[0.1, 0.2, 0.25],
    limit=2
)

print("🔍 搜索结果：")
for hit in hits:
    print(f"id={hit.id}, score={hit.score}, payload={hit.payload}")
