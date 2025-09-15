import numpy as np
from collections import defaultdict

# ========== DBSF 实现 ==========
def normalize_dbsf(scores):
    values = np.array(list(scores.values()))
    mu, sigma = values.mean(), values.std(ddof=1) if len(values) > 1 else (values.mean(), 1e-9)
    lower, upper = mu - 3*sigma, mu + 3*sigma
    norm_scores = {}
    for doc, s in scores.items():
        if upper == lower:
            ns = 0.5
        else:
            ns = (s - lower) / (upper - lower)
            ns = max(0.0, min(1.0, ns))
        norm_scores[doc] = ns
    return norm_scores

def dbsf_fusion(results_list):
    fused = defaultdict(float)
    for scores in results_list:
        norm_scores = normalize_dbsf(scores)
        for doc, ns in norm_scores.items():
            fused[doc] += ns
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

# ========== Min-Max + 加权融合 ==========
def normalize_minmax(scores):
    vals = list(scores.values())
    minv, maxv = min(vals), max(vals)
    return {doc: (s - minv) / (maxv - minv + 1e-9) for doc, s in scores.items()}

def weighted_fusion(results_list, weights=None):
    fused = defaultdict(float)
    if weights is None:
        weights = [1.0] * len(results_list)
    for scores, w in zip(results_list, weights):
        norm_scores = normalize_minmax(scores)
        for doc, ns in norm_scores.items():
            fused[doc] += w * ns
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

# ========== 构造不同分布的检索结果 ==========
# BM25 (0–30)
bm25 = {"doc1": 28.4, "doc2": 17.2, "doc3": 3.9, "doc4": 10.5}

# Dense embedding (0.2–0.9)
dense = {"doc1": 0.78, "doc2": 0.65, "doc3": 0.52, "doc4": 0.31}

# 行为特征 (很窄的分布 0.01–0.05)
ctr = {"doc1": 0.045, "doc2": 0.032, "doc3": 0.028, "doc4": 0.041}

# ========== 对比结果 ==========
print("=== Min-Max + Weighted (BM25=0.5, Dense=0.3, CTR=0.2) ===")
print(weighted_fusion([bm25, dense, ctr], weights=[0.3, 0.3, 0.3]))

print("\n=== DBSF Fusion ===")
print(dbsf_fusion([bm25, dense, ctr]))
