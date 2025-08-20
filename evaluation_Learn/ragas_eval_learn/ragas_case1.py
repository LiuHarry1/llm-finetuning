# https://docs.ragas.io/en/stable/getstarted/evals/#evaluation
import asyncio

from langchain_community.chat_models import ChatTongyi
from ragas import SingleTurnSample
from ragas.metrics import BleuScore

def bleu_score():

    test_data = {
        "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
        "reference": "The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."
    }
    metric = BleuScore()
    test_data = SingleTurnSample(**test_data)
    result = metric.single_turn_score(test_data)
    print(result)


def rouge_score():
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import RougeScore

    sample = SingleTurnSample(
        response="The Eiffel Tower is located in India.",
        reference="The Eiffel Tower is located in Paris."
    )

    scorer = RougeScore()
    await scorer.single_turn_ascore(sample)



async def AspectCritic_test():
    import os
    from ragas.llms import LangchainLLMWrapper

    llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
    evaluator_llm = LangchainLLMWrapper(llm)

    from ragas import SingleTurnSample
    from ragas.metrics import AspectCritic

    test_data = {
        "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    }

    metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate.")
    test_data = SingleTurnSample(**test_data)
    result = await metric.single_turn_ascore(test_data)
    print(result)

def evaluate_on_dataset():
    from datasets import load_dataset
    from ragas import EvaluationDataset
    from ragas.metrics import AspectCritic
    from ragas.llms import LangchainLLMWrapper

    llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
    evaluator_llm = LangchainLLMWrapper(llm)

    metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate.")

    eval_dataset = load_dataset("explodinggradients/earning_report_summary", split="train")
    eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
    eval_dataset = eval_dataset[:4]
    print("Features in dataset:", eval_dataset.features())
    print("Total samples in dataset:", len(eval_dataset))

    from ragas import evaluate

    results = evaluate(eval_dataset, metrics=[metric])
    print(results)
    print(results.to_pandas())

class RAG:
    def __init__(self):
        self.documents = []

    def load_documents(self, docs):
        """加载文档到 RAG 实例"""
        self.documents = docs

    def get_most_relevant_docs(self, query):
        """
        简单 mock 方式：根据关键词匹配返回最相关的文档
        这里直接用最简单的字符串包含判断
        """
        for doc in self.documents:
            for word in query.split():
                if word.lower() in doc.lower():
                    return [doc]  # 返回列表形式
        # 如果没匹配到，返回第一个文档
        return [self.documents[0]]

    def generate_answer(self, query, retrieved_docs):
        """
        简单 mock 生成答案：直接返回检索到的文档
        实际可调用 llm 来生成答案
        """
        # 这里假设 retrieved_docs 永远只有一个文档
        if retrieved_docs:
            return retrieved_docs[0]
        return "No answer found."

def rag_evaluation():

    llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")


    sample_docs = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
    ]

    # Initialize RAG instance
    rag = RAG()

    # Load documents
    rag.load_documents(sample_docs)

    # Query and retrieve the most relevant document
    query = "Who introduced the theory of relativity?"
    relevant_doc = rag.get_most_relevant_docs(query)

    # Generate an answer
    answer = rag.generate_answer(query, relevant_doc)

    print(f"Query: {query}")
    print(f"Relevant Document: {relevant_doc}")
    print(f"Answer: {answer}")

    sample_queries = [
        "Who introduced the theory of relativity?",
        "Who was the first computer programmer?",
        "What did Isaac Newton contribute to science?",
        "Who won two Nobel Prizes for research on radioactivity?",
        "What is the theory of evolution by natural selection?"
    ]

    expected_responses = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
    ]

    dataset = []

    for query, reference in zip(sample_queries, expected_responses):
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": relevant_docs,
                "response": response,
                "reference": reference
            }
        )

    from ragas import EvaluationDataset
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper

    evaluator_llm = LangchainLLMWrapper(llm)
    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, AnswerCorrectness, ContextRecall

    result = evaluate(dataset=evaluation_dataset, metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
                      llm=evaluator_llm)
    print(result)

    # {'context_recall': 1.0000, 'faithfulness': 0.8571, 'factual_correctness': 0.7280}


if __name__ == '__main__':
    # bleu_score()
    # asyncio.run(AspectCritic_test())
    evaluate_on_dataset()
    # rag_evaluation()