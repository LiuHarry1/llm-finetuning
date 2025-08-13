import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# 1. 加载模型和分词器
model_name = '/Users/harry/Documents/apps/ml/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 编写一个函数，计算句子embedding（mean pooling）
def encode(texts):
    # 支持输入列表或单句字符串
    if isinstance(texts, str):
        texts = [texts]
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    # model_output.last_hidden_state shape: (batch_size, seq_len, hidden_size)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    # Mean Pooling
    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return embeddings

# 3. 准备数据
chunks = [
    "BM25 is a probabilistic information retrieval algorithm.",
    "Neural networks can learn semantic representations of text.",
    "Quantum decoherence causes classical behavior in quantum systems.",
    "Retrieval-augmented generation combines search with language models.",
    "Cosine similarity is commonly used in vector search systems."
]

query = "How does quantum decoherence work?"

# 4. 计算 embeddings
chunk_embeddings = encode(chunks)  # shape (5, hidden_size)
query_embedding = encode(query)    # shape (1, hidden_size)

# 5. 计算余弦相似度
cosine_scores = F.cosine_similarity(query_embedding, chunk_embeddings)
# cosine_scores shape: (5,)

# 6. 打印top3最相似chunk
top_k = 3
top_scores, top_indices = torch.topk(cosine_scores, k=top_k)
print(f"\nQuery: {query}\nTop {top_k} relevant chunks:")
for score, idx in zip(top_scores, top_indices):
    print(f"Score: {score.item():.4f} | Chunk: {chunks[idx]}")

