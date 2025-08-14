from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI as OpenAiLlm
import tiktoken
import os
from openai import OpenAI


chat_history = [
    ChatMessage(role="user", content="What is LlamaIndex?"),
    ChatMessage(
        role="assistant",
        content="LlamaaIndex is the leading data framework for building LLM applications",
    ),
    ChatMessage(role="user", content="Can you give me some more details?"),
    ChatMessage(
        role="assistant",
        content="""LlamaIndex is a framework for building context-augmented LLM applications. Context augmentation refers to any use case that applies LLMs on top of your private or domain-specific data. Some popular use cases include the following: 
        Question-Answering Chatbots (commonly referred to as RAG systems, which stands for "Retrieval-Augmented Generation"), Document Understanding and Extraction, Autonomous Agents that can perform research and take actions
        LlamaIndex provides the tools to build any of these above use cases from prototype to production. The tools allow you to both ingest/process this data and implement complex query workflows combining data access with LLM prompting.""",
    ),
]

# model = "gpt-4-0125-preview"
summarizer_llm = OpenAiLlm(model_name=model, max_tokens=256)
summarizer_llm = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# tokenizer_fn = tiktoken.encoding_for_model(model).encode
memory = ChatSummaryMemoryBuffer.from_defaults(
    chat_history=chat_history,
    llm=summarizer_llm,
    token_limit=2,
    # tokenizer_fn=tokenizer_fn,
)

history = memory.get()

print(history)