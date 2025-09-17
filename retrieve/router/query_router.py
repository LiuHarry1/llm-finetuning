import logging
import os
import sys

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi

# https://developers.llamaindex.ai/python/examples/retrievers/router_retriever

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    SimpleKeywordTableIndex,
)
from llama_index.core import SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# initialize LLM + splitter

load_dotenv()

# 初始化 ChatTongyi，开启流式
llm = ChatTongyi(model="qwen-plus", temperature=0.7, api_key=os.getenv("DASHSCOPE_API_KEY"), streaming=False )

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# define
summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

list_retriever = summary_index.as_retriever()
vector_retriever = vector_index.as_retriever()
keyword_retriever = keyword_index.as_retriever()


from llama_index.core.tools import RetrieverTool

list_tool = RetrieverTool.from_defaults(
    retriever=list_retriever,
    description=(
        "Will retrieve all context from Paul Graham's essay on What I Worked"
        " On. Don't use if the question only requires more specific context."
    ),
)
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_retriever,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)
keyword_tool = RetrieverTool.from_defaults(
    retriever=keyword_retriever,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On (using entities mentioned in query)"
    ),
)

from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.selectors import (
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.response.notebook_utils import display_source_node

retriever = RouterRetriever(
    selector=PydanticSingleSelector.from_defaults(llm=llm),
    retriever_tools=[
        list_tool,
        vector_tool,
    ],
)

# will retrieve all context from the author's life
nodes = retriever.retrieve(
    "Can you give me all the context regarding the author's life?"
)
for node in nodes:
    display_source_node(node)


# nodes = retriever.retrieve("What did Paul Graham do after RISD?")
# for node in nodes:
#     display_source_node(node)