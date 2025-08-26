import os
import traceback

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from mem0 import Memory, MemoryClient
from mem0.configs.base import MemoryConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.graphs.configs import GraphStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

# https://blog.futuresmart.ai/integrating-mem0-with-langchain
# https://docs.mem0.ai/integrations/langchain

custom_prompt = """
Please only extract entities containing patient health information, appointment details, and user information. 
Here are some few shot examples:

Input: Hi.
Output: {{"facts" : []}}

Input: The weather is nice today.
Output: {{"facts" : []}}

Input: I have a headache and would like to schedule an appointment.
Output: {{"facts" : ["Patient reports headache", "Wants to schedule an appointment"]}}

Input: My name is Jane Smith, and I need to reschedule my appointment for next Tuesday.
Output: {{"facts" : ["Patient name: Jane Smith", "Wants to reschedule appointment", "Original appointment: next Tuesday"]}}

Input: I have diabetes and my blood sugar is high.
Output: {{"facts" : ["Patient has diabetes", "Reports high blood sugar"]}}

Return the facts and patient information in a json format as shown above.
"""

load_dotenv()
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")

llm = ChatTongyi(model="qwen-plus", api_key=TONGYI_API_KEY)
embeder = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key = TONGYI_API_KEY)

# 1. 配置 Memory
config = MemoryConfig( llm = LlmConfig( provider="langchain", config={"model":llm }, ),
    embedder = EmbedderConfig( provider = "langchain", config= { "model":embeder} ),
    vector_store = VectorStoreConfig(provider = "qdrant",
                                     config={
                                         "host": "localhost",
                                         "port": 6333,
                                         "collection_name": "memory_vectors",
                                         "embedding_model_dims": 1536,
                                     }
                                     ),
    custom_fact_extraction_prompt = custom_prompt

    # graph_store=  GraphStoreConfig(provider = "neo4j",
    #                                    config= {
    #                                     "url": "bolt://localhost:7687",
    #                                     "username": "neo4j",
    #                                     "password": "myhome1234"
    #
    #                                     }
    #             ),

    )


class AIHealthcareSupport:
    def __init__(self, config):
        self.memory  = Memory(config=config)
        self.app_id = "app-1"
        self.model = llm

    def ask(self, question, user_id=None):
        memories = self.search_memory(question, user_id=user_id)
        context = self.convert_to_facts(memories['results'])

        print("context:",context)
        messages = [
            SystemMessage(content=f"""You are a helpful healthcare support assistant. 
            Use the provided context to personalize your responses and remember user health information 
            and past interactions. {context}"""),
            HumanMessage(content=question)
        ]

        response = self.model.invoke(messages)

        # Store the interaction in memory
        self.add_memory(question, response.content, user_id=user_id)
        return {"messages": [response.content]}

    def add_memory(self, question, response, user_id=None):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        self.memory.add(messages, user_id=user_id, metadata={"app_id": self.app_id})

    def get_memories(self, user_id=None):
        return self.memory.get_all(user_id=user_id)

    def search_memory(self, query, user_id=None):
        related_memories = self.memory.search(query, user_id=user_id)
        return related_memories

    def convert_to_facts(self, memories):

        if not memories:
            return ""
        output_lines = []
        output_lines.append("# These are the most relevant facts and their valid date ranges")
        output_lines.append("# format: FACT (Date range: from - to)")
        output_lines.append("<FACTS>")

        # 添加每个记忆项
        for memory in memories:
            content = memory.get('memory', '')
            created_at = memory.get('created_at')
            if memory.get('updated_at') and memory.get('updated_at')!=None:
                created_at = memory.get('updated_at')
            expiration_date = "present"
            if memory.get("expiration_date") and memory.get("expiration_date") != None:
                expiration_date = memory.get("expiration_date")

            # 格式化时间范围
            if created_at:
                time_range = f"({created_at} - {expiration_date})"
            else:
                time_range = "(unknown date range)"

            output_lines.append(f"  - {content} {time_range}")

        output_lines.append("</FACTS>")

        return "\n".join(output_lines)

def test1():
    # Initialize the AIHealthcareSupport bot
    ai_support = AIHealthcareSupport(config)
    # User ID for interaction
    user_id = "Harry"

    # Interacting with the bot
    print("Interacting with AI Healthcare Support:\n")

    # Example interactions
    questions = [
        "I have a family history of diabetes; how can I reduce my risk?",  # Preventive care inquiry
        "Can you recommend any specific dietary changes?",  # Focusing on diet
        "I have a headache and would like to schedule an appointment."
    ]

    # Loop through each question, ask the bot, and print responses
    for question in questions:
        response = ai_support.ask(question, user_id=user_id)
        print(f"User: {question}")
        print(f"AI: {response['messages'][0]}\n")

    # Retrieve and display memories associated with the user
    memories = ai_support.get_memories(user_id=user_id)
    print("All Memories:")
    for memory in memories['results']:
        print(f"- {memory}")

def chatbot():
    user_id, thread_id = "Harry", "25f570725f6a4233ad8942d9d1c6cc79"

    ai_support = AIHealthcareSupport(config)
    while True:
        try:
            user_input = input("🧑 User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            # print("user input", user_input)
            response = ai_support.ask(user_input, user_id)
            print(f"🤖 Assistant: {response['messages'][0]}")
        except Exception as e:

            print("发生错误:")
            traceback.print_exc()
            break



if __name__ == '__main__':

    # test1()
    chatbot()
    # memory = Memory(config=config)
    # memories = memory.search("I have a family history of diabetes; how can I reduce my risk?", user_id="Harry")
    # aIHealthcareSupport  = AIHealthcareSupport(config)
    # context = aIHealthcareSupport.convert_to_facts(memories['results'])
    # print(context)

