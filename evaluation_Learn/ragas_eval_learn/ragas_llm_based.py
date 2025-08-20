from langchain_community.chat_models import ChatTongyi
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextEntityRecall
import asyncio

def context_entity_recall():

    sample = SingleTurnSample(
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["The Eiffel Tower is located in Paris."],
    )

    chat_llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
    # 初始化语言模型
    evaluator_llm = LangchainLLMWrapper(chat_llm)

    scorer = ContextEntityRecall(llm=evaluator_llm)

    result = asyncio.run(scorer.single_turn_ascore(sample))

    print(result)

def llm_based_context_recall():
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import LLMContextRecall

    chat_llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
    evaluator_llm = LangchainLLMWrapper(chat_llm)

    sample = SingleTurnSample(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["Paris is the capital of France."],
    )

    context_recall = LLMContextRecall(llm=evaluator_llm)
    result = asyncio.run( context_recall.single_turn_ascore(sample))
    print(result)

def factual_correctness_test():
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._factual_correctness import FactualCorrectness
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import LLMContextRecall

    chat_llm = ChatTongyi(model="qwen-plus", api_key="sk-f256c03643e9491fb1ebc278dd958c2d")
    evaluator_llm = LangchainLLMWrapper(chat_llm)

    sample = SingleTurnSample(
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris. I has a height of 1000ft."
    )

    scorer = FactualCorrectness(llm=evaluator_llm)
    await scorer.single_turn_ascore(sample)


if __name__ == '__main__':
    # context_entity_recall()
    # none_llm_based_context_recall()
    llm_based_context_recall()