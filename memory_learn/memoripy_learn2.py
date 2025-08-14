import numpy as np
from langchain_openai import OpenAIEmbeddings
from memoripy import MemoryManager, JSONStorage, EmbeddingModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from pydantic import BaseModel, Field

from memoripy.implemented_models import OpenAIChatModel, OllamaEmbeddingModel, ChatModel, ChatOpenAI


class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")



class QwenChatModel(ChatModel):
    def __init__(self, api_key: str, model_name: str = "qwen-plus"):
        # 用 OpenAI API 兼容模式调用 Qwen
        self.api_key = api_key
        self.model_name = model_name
        self.llm = ChatOpenAI(
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key=api_key,
            model_name=model_name
        )
        self.parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)
        self.prompt_template = PromptTemplate(
            template=(
                "Extract key concepts from the following text in a concise, context-specific manner. "
                "Include only the most highly relevant and specific core concepts that best capture the text's meaning. "
                "Return nothing but the JSON string.\n"
                "{format_instructions}\n{text}"
            ),
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def invoke(self, messages: list) -> str:
        response = self.llm.invoke(messages)
        return str(response.content)

    def extract_concepts(self, text: str) -> list[str]:
        chain = self.prompt_template | self.llm | self.parser
        response = chain.invoke({"text": text})
        concepts = response.get("concepts", [])
        print(f"Concepts extracted: {concepts}")
        return concepts

class QwenEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key: str, model_name="text-embedding-v4", dimensions=1024):
        self.api_key = api_key
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def get_embedding(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        embedding = resp.data[0].embedding
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        return np.array(embedding)

    def initialize_embedding_dimension(self) -> int:
        return self.dimensions


def test_qwen(api_key: str):
    """测试 Qwen 模型的聊天、概念提取和向量生成"""
    import numpy as np

    # 初始化模型
    chat_model = QwenChatModel(api_key=api_key, model_name="qwen-plus")
    embedding_model = QwenEmbeddingModel(api_key=api_key, model_name="text-embedding-v1")

    # 测试聊天
    print("\n=== 测试 Qwen Chat ===")
    messages = [{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    reply = chat_model.invoke(messages)
    print("模型回复：", reply)

    # 测试概念提取
    print("\n=== 测试概念提取 ===")
    text = "苹果公司发布了新一代 iPhone，采用了全新的 A18 芯片和改进的摄像系统。"
    concepts = chat_model.extract_concepts(text)
    print("提取到的概念：", concepts)

    # 测试 Embedding
    print("\n=== 测试 Qwen Embedding ===")
    text = "人工智能正在改变世界"
    vector = embedding_model.get_embedding(text)
    print("向量维度：", len(vector))
    print("向量前 5 个值：", np.round(vector[:5], 4))  # 取前 5 个值并四舍五入


# 直接运行测试（替换为你的 DashScope API Key）
if __name__ == "__main__":
    test_qwen(api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
