from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

dashScopeEmbeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key = "sk-f256c03643e9491fb1ebc278dd958c2d")

llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")

generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = LangchainEmbeddingsWrapper(dashScopeEmbeddings)

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

result = dataset.to_pandas()

# 转换为 pandas DataFrame
result = dataset.to_pandas()
print(result)
# 保存为 CSV 文件
result.to_csv("ragas_eval_results.csv", index=False, encoding="utf-8")

# 保存为 Excel 文件
result.to_excel("ragas_eval_results.xlsx", index=False)

# 保存为 JSON 文件
result.to_json("ragas_eval_results.json", orient="records", force_ascii=False, indent=2)

