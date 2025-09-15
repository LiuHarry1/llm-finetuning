import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import SparseVector
from typing import List, Optional

# https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridSearchEngine:
    def __init__(self, qdrant_url: str, qdrant_api_key: Optional[str], collection_name: str, model_path: str,
                 docs: List[str]):
        """
        初始化混合搜索引擎
        Args:
            qdrant_url: Qdrant服务器地址
            qdrant_api_key: Qdrant API密钥（可选）
            collection_name: Qdrant集合名称
            model_path: 本地嵌入模型路径
            docs: 文档列表
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.model_path = model_path
        self.docs = docs

        # 初始化Qdrant客户端
        self.client = self._init_qdrant_client()

        # 加载模型和分词器
        self.tokenizer, self.model = self._load_model_and_tokenizer()

        # 初始化BM25
        self.bm25 = self._init_bm25()

    def _init_qdrant_client(self) -> QdrantClient:
        """初始化Qdrant客户端"""
        try:
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=60  # 设置超时时间
            )
            # 简单检查连接是否正常
            client.get_collections()
            logger.info("Successfully connected to Qdrant")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _load_model_and_tokenizer(self):
        """加载本地模型和分词器"""
        try:
            logger.info(f"Loading model and tokenizer from {self.model_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModel.from_pretrained(self.model_path)
            logger.info("Model and tokenizer loaded successfully")
            return tokenizer, model
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise

    def _init_bm25(self):
        """初始化BM25"""
        tokenized_docs = [doc.lower().split() for doc in self.docs]
        return BM25Okapi(tokenized_docs)

    def encode(self, texts: List[str]) -> List[List[float]]:
        """将文本列表编码为嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]

        try:
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            raise

    def bm25_vectorize(self, doc: str) -> SparseVector:
        """生成BM25稀疏向量"""
        tokens = doc.lower().split()
        scores = self.bm25.get_scores(tokens)
        indices = [i for i, s in enumerate(scores) if s > 0]
        values = [float(scores[i]) for i in indices]
        return SparseVector(indices=indices, values=values)

    def create_collection(self):
        """创建Qdrant集合"""
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,  # 模型输出维度
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                }
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            # 可以根据具体异常决定是抛出还是处理

    def index_documents(self):
        """索引文档到Qdrant"""
        try:
            dense_embeddings = self.encode(self.docs)
            points = []
            for idx, (doc, dense_vec) in enumerate(zip(self.docs, dense_embeddings), start=1):
                sparse_vec = self.bm25_vectorize(doc)
                points.append(
                    models.PointStruct(
                        id=idx,
                        vector={"dense": dense_vec, "sparse": sparse_vec},
                        payload={"text": doc},
                    )
                )

            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # 等待操作完成
            )
            logger.info(f"Documents indexed successfully. Operation info: {operation_info}")
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise

    def search(self, query: str, limit: int = 5) -> List:
        """执行混合搜索"""
        try:
            # 生成查询的稠密向量
            query_dense = self.encode([query])[0]  # encode返回列表的列表
            # 生成查询的稀疏向量
            query_sparse_vector = self.bm25_vectorize(query)

            # 执行搜索
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_sparse_vector,
                        using="sparse",
                        limit=limit
                    ),
                    models.Prefetch(
                        query=query_dense,
                        using="dense",
                        limit=limit
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.DBSF),
                limit=limit,
            )
            logger.info(f"Search for '{query}' completed successfully")
            return results.points
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise


# 使用示例
if __name__ == "__main__":
    # 从环境变量获取配置
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bm25_dense_local")
    MODEL_PATH = os.getenv("MODEL_PATH", "/Users/harry/Documents/apps/ml/all-MiniLM-L6-v2")

    # 文档数据（生产环境中应从数据库或文件中读取）
    DOCUMENTS = [
        "Cats are wonderful pets.",
        "Dogs are loyal animals.",
        "Birds can fly in the sky.",
        "Fish swim in the water.",
        "Retrieval-augmented generation combines search with language models.",
    ]

    try:
        # 初始化搜索引擎
        search_engine = HybridSearchEngine(
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            model_path=MODEL_PATH,
            docs=DOCUMENTS
        )

        # 创建集合（如果尚未存在）
        search_engine.create_collection()

        # 索引文档
        search_engine.index_documents()

        # 执行查询
        query = "Tell me about animals that are good friends"
        results = search_engine.search(query, limit=5)

        # 输出结果
        print(f"\nQuery: {query}\nTop results:")
        for hit in results:
            print(f"Score: {hit.score:.4f} | Text: {hit.payload['text']}")

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")