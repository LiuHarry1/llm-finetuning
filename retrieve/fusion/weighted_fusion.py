def normalize_scores(results):
    """
    归一化分数到 0~1
    results: [(doc_id, score), ...]
    """
    if not results:
        return {}
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:  # 避免除零
        return {doc_id: 1.0 for doc_id, _ in results}
    return {doc_id: (score - min_s) / (max_s - min_s) for doc_id, score in results}


def weighted_fusion(bm25_results, vector_results, w_bm25=0.5, w_vector=0.5):
    """
    bm25_results: [(doc_id, score), ...]
    vector_results: [(doc_id, score), ...]
    w_bm25, w_vector: 权重，和为 1
    """
    # 归一化
    bm25_norm = normalize_scores(bm25_results)
    vector_norm = normalize_scores(vector_results)

    # 加权融合
    scores = {}
    all_docs = set(bm25_norm) | set(vector_norm)
    for doc_id in all_docs:
        score = w_bm25 * bm25_norm.get(doc_id, 0) + w_vector * vector_norm.get(doc_id, 0)
        scores[doc_id] = score

    # 排序
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# 示例数据
bm25_results = [("d1", 12.0), ("d2", 11.5), ("d3", 10.0), ("d4", 9.0)]
vector_results = [("d3", 0.88), ("d2", 0.85), ("d5", 0.83), ("d1", 0.80)]

# 融合，BM25 占 0.6 权重，向量占 0.4 权重
fused_results = weighted_fusion(bm25_results, vector_results, w_bm25=0.6, w_vector=0.4)
print(fused_results)
