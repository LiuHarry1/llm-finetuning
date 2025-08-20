# https://docs.ragas.io/en/stable/getstarted/evals/#evaluation
import asyncio

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

def string_simiarity():
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics._string import NonLLMStringSimilarity

    sample = SingleTurnSample(
        response="The Eiffel Tower is located in India.",
        reference="The Eiffel Tower is located in Paris."
    )

    scorer = NonLLMStringSimilarity()
    await scorer.single_turn_ascore(sample)

def  string_presence():
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import StringPresence

    sample = SingleTurnSample(
        response="The Eiffel Tower is located in India.",
        reference="Eiffel Tower"
    )
    scorer = StringPresence()
    await scorer.single_turn_ascore(sample)


if __name__ == '__main__':
    bleu_score()
