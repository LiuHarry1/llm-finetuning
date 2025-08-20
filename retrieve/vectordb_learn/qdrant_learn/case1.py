from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest, MatchText, VectorParams, Distance
import json

# 连接到 Qdrant
client = QdrantClient("localhost", port=6333)

# 先创建集合并插入一些数据
client.recreate_collection(
    collection_name="bm25_demo",
    vectors_config=VectorParams(size=4, distance=Distance.DOT)  # 使用 DOT 确保兼容性
)

documents = [
    {
        "id": 1,
        "vector": [0.1, 0.2, 0.3, 0.4],
        "payload": {
            "text": "机器学习是人工智能的重要分支",
            "category": "AI",
            "author": "张三"
        }
    },
    {
        "id": 2,
        "vector": [0.5, 0.6, 0.7, 0.8],
        "payload": {
            "text": "深度学习使用神经网络进行特征学习",
            "category": "AI",
            "author": "李四"
        }
    },
    {
        "id": 3,
        "vector": [0.9, 1.0, 0.1, 0.2],
        "payload": {
            "text": "自然语言处理让计算机理解人类语言",
            "category": "NLP",
            "author": "王五"
        }
    }
]

client.upsert(
    collection_name="bm25_demo",
    points=documents
)

print("数据插入完成")


# 方法1：使用 SearchRequest 进行 BM25 搜索（推荐）
def bm25_search_with_search_request():
    """使用 SearchRequest 进行 BM25 搜索"""
    search_request = SearchRequest(
        query=MatchText(text="机器学习"),  # 使用 MatchText 进行文本搜索
        limit=3,
        with_payload=True
    )

    results = client.search(
        collection_name="bm25_demo",
        search_request=search_request
    )

    print("🔍 使用 SearchRequest 的 BM25 搜索结果:")
    for result in results:
        print(f"ID: {result.id}, Score: {result.score:.4f}")
        print(f"Text: {result.payload['text']}")
        print("---")
    return results


# 方法2：使用 query_filter 进行文本匹配
def bm25_search_with_query_filter():
    """使用 query_filter 进行文本搜索"""
    results = client.query_points(
        collection_name="bm25_demo",
        query_filter=Filter(
            must=[FieldCondition(key="text", match=MatchText(text="神经网络"))]
        ),
        limit=3,
        with_payload=True
    ).points

    print("🔍 使用 query_filter 的文本搜索结果:")
    for result in results:
        print(f"ID: {result.id}")
        print(f"Text: {result.payload['text']}")
        print("---")
    return results




try:
    results2 = bm25_search_with_query_filter()
except Exception as e:
    print(f"query_filter 方法失败: {e}")


# 方法4：如果以上方法都失败，使用更基础的过滤
def basic_text_search():
    """基础文本搜索"""
    # 获取所有点然后本地过滤（不推荐用于大量数据）
    all_points = client.scroll(
        collection_name="bm25_demo",
        limit=100,
        with_payload=True
    )[0]

    search_term = "学习"
    filtered_results = [
        point for point in all_points
        if search_term in point.payload.get('text', '')
    ]

    print(f"🔍 本地过滤搜索结果 (关键词: '{search_term}'):")
    for result in filtered_results[:3]:  # 显示前3个结果
        print(f"ID: {result.id}")
        print(f"Text: {result.payload['text']}")
        print("---")
    return filtered_results


# 运行基础搜索
basic_text_search()