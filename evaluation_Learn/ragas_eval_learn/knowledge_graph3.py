from langchain_community.chat_models import ChatTongyi
from ragas.testset.graph import Node, Relationship, KnowledgeGraph
from ragas.testset.transforms.extractors import NERExtractor, SummaryExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.transforms.relationship_builders import CosineSimilarityBuilder
from ragas.llms import LangchainLLMWrapper


chat_llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
# 初始化语言模型
llm = LangchainLLMWrapper(chat_llm)

# 加载文档
documents = [...]  # 你的文档列表

# 实体和主题抽取
ner_extractor = NERExtractor(llm=llm)
theme_extractor = SummaryExtractor(llm=llm)

# 文档分割
splitter = HeadlineSplitter(min_tokens=500)

# 关系构建
cosine_sim_builder = CosineSimilarityBuilder(
    property_name="summary_embedding",
    new_property_name="summary_similarity",
    threshold=0.7
)

# 创建知识图谱
kg = KnowledgeGraph()

# 添加节点和关系
for doc in documents:
    node = Node(properties={"page_content": doc})
    kg.add(node)
    # 进一步处理，抽取实体、主题，建立关系等

# 生成测试集



# testset = generate_testset(kg)
