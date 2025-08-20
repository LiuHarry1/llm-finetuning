from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SearchRequest, Filter, FieldCondition, MatchValue
import numpy as np


def modern_qdrant_example():
    # 连接到 Qdrant
    client = QdrantClient("localhost", port=6333)

    collection_name = "modern_collection"

    # 1. 检查集合是否存在，如果存在则删除
    existing_collections = client.get_collections().collections
    collection_names = [col.name for col in existing_collections]

    if collection_name in collection_names:
        print(f"删除已存在的集合: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    # 2. 创建新集合（使用新的 API）
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )

    print(f"创建集合: {collection_name}")

    # 3. 插入数据
    client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": 1,
                "vector": [0.9, 0.1, 0.8, 0.2],
                "payload": {"text": "apple", "category": "fruit"}
            },
            {
                "id": 2,
                "vector": [0.2, 0.9, 0.3, 0.7],
                "payload": {"text": "banana", "category": "fruit"}
            },
            {
                "id": 3,
                "vector": [0.7, 0.3, 0.1, 0.9],
                "payload": {"text": "carrot", "category": "vegetable"}
            }
        ]
    )

    print("数据插入完成")

    # 4. 查询示例
    query_vector = [0.8, 0.2, 0.7, 0.3]

    # 使用 query_points 方法（推荐）
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3,
        with_payload=True
    ).points

    print("\n🔍 查询结果:")
    for result in results:
        print(f"ID: {result.id}, Score: {result.score:.4f}, Text: {result.payload['text']}")

    return results


def check_collection_exists(client, collection_name):
    """检查集合是否存在的辅助函数"""
    try:
        client.get_collection(collection_name=collection_name)
        return True
    except Exception:
        return False


def safe_recreate_collection(client, collection_name, vectors_config):
    """安全的集合重建函数（替代 recreate_collection）"""
    # 检查集合是否存在
    if check_collection_exists(client, collection_name):
        print(f"删除已存在的集合: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    # 创建新集合
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config
    )
    print(f"创建新集合: {collection_name}")


# 更简洁的使用方式
def simple_example():
    client = QdrantClient("localhost", port=6333)
    collection_name = "test_simple"

    # 使用安全的集合重建
    safe_recreate_collection(
        client=client,
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.DOT)
    )

    # 插入数据
    vectors = [
        [1.0, 0.0, 0.5, 0.3],  # vector 1
        [0.0, 1.0, 0.3, 0.8],  # vector 2
        [0.5, 0.5, 1.0, 0.0]  # vector 3
    ]

    client.upsert(
        collection_name=collection_name,
        points=[
            {"id": 1, "vector": vectors[0], "payload": {"text": "vector 1"}},
            {"id": 2, "vector": vectors[1], "payload": {"text": "vector 2"}},
            {"id": 3, "vector": vectors[2], "payload": {"text": "vector 3"}}
        ]
    )

    # 查询 - 使用新的 query_points API
    query_vector = [0.9, 0.1, 0.4, 0.2]
    hits = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=2
    ).points

    print("🔍 搜索结果：")
    for hit in hits:
        print(f"id={hit.id}, score={hit.score:.6f}, payload={hit.payload}")


def get_qdrant_health_status(client):
    """检查 Qdrant 服务状态"""
    try:
        health = client.get_health()
        return f"Status: {health.status}, Version: {health.version}"
    except Exception as e:
        return f"Health check failed: {e}"

def list_all_collections(client):
    """列出所有集合"""
    collections = client.get_collections().collections
    return [col.name for col in collections]

def collection_info(client, collection_name):
    """获取集合详细信息"""
    try:
        return client.get_collection(collection_name=collection_name)
    except Exception as e:
        return f"Collection not found: {e}"

if __name__ == "__main__":
    print("运行完整示例...")
    modern_qdrant_example()

    print("\n" + "=" * 50 + "\n")

    print("运行简洁示例...")
    simple_example()