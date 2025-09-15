import numpy as np
from typing import List, Dict, Any


def normalize_scores(scores: List[float]) -> List[float]:
    """将分数列表归一化到[0,1]范围"""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5] * len(scores)  # 所有分数相同时返回中间值
    return [(s - min_score) / (max_score - min_score) for s in scores]


def weighted_fusion(bm25_results: List[Dict], vector_results: List[Dict],
                    bm25_weight: float = 0.5, vector_weight: float = 0.5):
    """
    加权融合BM25和向量搜索结果

    Args:
        bm25_results: BM25搜索结果列表，每个元素包含 'id', 'score', 'payload'
        vector_results: 向量搜索结果列表，每个元素包含 'id', 'score', 'payload'
        bm25_weight: BM25结果的权重
        vector_weight: 向量结果的权重

    Returns:
        融合后的排序结果列表
    """
    # 提取分数并归一化
    bm25_scores = [result['score'] for result in bm25_results]
    vector_scores = [result['score'] for result in vector_results]

    normalized_bm25 = normalize_scores(bm25_scores)
    normalized_vector = normalize_scores(vector_scores)

    # 创建结果字典，方便按ID查找
    bm25_dict = {result['id']: {'score': norm_score, 'payload': result['payload']}
                 for result, norm_score in zip(bm25_results, normalized_bm25)}

    vector_dict = {result['id']: {'score': norm_score, 'payload': result['payload']}
                   for result, norm_score in zip(vector_results, normalized_vector)}

    # 获取所有唯一的文档ID
    all_ids = set(bm25_dict.keys()) | set(vector_dict.keys())

    # 计算融合分数
    fused_results = []
    for doc_id in all_ids:
        bm25_score = bm25_dict.get(doc_id, {'score': 0})['score']
        vector_score = vector_dict.get(doc_id, {'score': 0})['score']

        # 加权融合
        fused_score = (bm25_weight * bm25_score + vector_weight * vector_score)

        # 获取payload（优先使用BM25的payload，因为它们通常包含原始文本）
        payload = bm25_dict.get(doc_id, vector_dict.get(doc_id, {}))['payload']

        fused_results.append({
            'id': doc_id,
            'fused_score': fused_score,
            'bm25_score': bm25_score,
            'vector_score': vector_score,
            'payload': payload
        })

    # 按融合分数降序排序
    fused_results.sort(key=lambda x: x['fused_score'], reverse=True)

    return fused_results


# 使用示例
def example_usage():
    # 模拟BM25搜索结果
    bm25_results = [
        {'id': 1, 'score': 8.5, 'payload': {'text': 'Cats are wonderful pets'}},
        {'id': 2, 'score': 7.2, 'payload': {'text': 'Dogs are loyal animals'}},
        {'id': 3, 'score': 6.8, 'payload': {'text': 'Birds can fly in the sky'}},
        {'id': 4, 'score': 5.1, 'payload': {'text': 'Fish swim in the water'}}
    ]

    # 模拟向量搜索结果
    vector_results = [
        {'id': 2, 'score': 0.92, 'payload': {'text': 'Dogs are loyal animals'}},
        {'id': 1, 'score': 0.85, 'payload': {'text': 'Cats are wonderful pets'}},
        {'id': 5, 'score': 0.78, 'payload': {'text': 'Retrieval-augmented generation'}},
        {'id': 3, 'score': 0.65, 'payload': {'text': 'Birds can fly in the sky'}}
    ]

    # 进行加权融合
    fused_results = weighted_fusion(
        bm25_results,
        vector_results,
        bm25_weight=0.4,  # BM25权重
        vector_weight=0.6  # 向量搜索权重
    )

    # 打印结果
    print("融合搜索结果:")
    print("-" * 80)
    for i, result in enumerate(fused_results, 1):
        print(f"{i}. ID: {result['id']}")
        print(f"   融合分数: {result['fused_score']:.4f}")
        print(f"   BM25分数: {result['bm25_score']:.4f}")
        print(f"   向量分数: {result['vector_score']:.4f}")
        print(f"   内容: {result['payload']['text']}")
        print()

        # 使用自适应权重
    bm25_weight, vector_weight = adaptive_weighting("your query here")
    fused_results = weighted_fusion(bm25_results, vector_results, bm25_weight, vector_weight)





def adaptive_weighting(query: str, default_bm25_weight: float = 0.4):
    """
    根据查询特性自适应调整权重
    """
    # 简单的启发式规则
    query_tokens = query.lower().split()

    # 如果查询很短（可能是关键词搜索），增加BM25权重
    if len(query_tokens) <= 2:
        bm25_weight = 0.6
        vector_weight = 0.4
    # 如果查询很长（可能是语义搜索），增加向量权重
    elif len(query_tokens) >= 8:
        bm25_weight = 0.3
        vector_weight = 0.7
    else:
        bm25_weight = default_bm25_weight
        vector_weight = 1 - default_bm25_weight

    return bm25_weight, vector_weight

if __name__ == "__main__":
    example_usage()

