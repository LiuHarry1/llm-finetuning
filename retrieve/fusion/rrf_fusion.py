def rrf_fusion_with_scores(rank_lists, k=60, score_weight=0.5):
    """
    rank_lists: [ [(doc_id, score), (doc_id, score), ...],  ... ]
        - 每个列表对应一个搜索器的结果
        - doc_id: 文档 ID
        - score: 该搜索器的原始得分（BM25 / 向量相似度等）
    k: RRF 平滑常数（越大排名影响越弱，默认 60）
    score_weight: 原始得分的权重比例（0~1）
    """
    scores = {}
    for result_list in rank_lists:
        for rank, (doc_id, raw_score) in enumerate(result_list, start=1):
            rrf_score = 1 / (k + rank)              # RRF 部分
            combined_score = rrf_score + score_weight * raw_score
            scores[doc_id] = scores.get(doc_id, 0) + combined_score

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# 示例：BM25 与向量搜索的结果 (doc_id, score)
bm25_results = [
    ("d1", 12.0), ("d2", 11.5), ("d3", 10.0), ("d4", 9.0)
]
vector_results = [
    ("d3", 0.88), ("d2", 0.85), ("d5", 0.83), ("d1", 0.80)
]

# 融合
fused_results = rrf_fusion_with_scores(
    [bm25_results, vector_results],
    k=60,
    score_weight=0.01  # BM25/向量分数范围不同，这里缩小权重
)

print(fused_results)
