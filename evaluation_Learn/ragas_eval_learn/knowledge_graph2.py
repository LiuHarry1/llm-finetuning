# 1. 创建图
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship

kg = KnowledgeGraph()

# 2. 创建节点
doc_node = Node(type=NodeType.DOCUMENT, properties={"title": "公司简介"})
chunk_node = Node(type=NodeType.CHUNK, properties={"text": "阿里巴巴是一家互联网公司"})

# 3. 创建关系
rel = Relationship(
    type="contains",
    source=doc_node,
    target=chunk_node
)

# 4. 添加到图
kg.add(doc_node)
kg.add(chunk_node)
kg.add(rel)

print(kg)
# 输出: KnowledgeGraph(nodes: 2, relationships: 1)
