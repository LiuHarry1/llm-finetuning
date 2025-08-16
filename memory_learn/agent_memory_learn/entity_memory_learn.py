from langchain.memory import ConversationEntityMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(temperature=0)

# 初始化实体记忆
entity_memory = ConversationEntityMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=entity_memory
)

print(conversation.predict(input="帮我看看项目 Alpha 最近有没有 Bug"))
print(conversation.predict(input="我主要关心支付模块"))
print(conversation.predict(input="那有没有高优先级的？"))
