import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


load_dotenv()

llm = ChatTongyi( model="qwen-plus", api_key=os.getenv("TONGYI_API_KEY"))


# 初始 agent 的 system prompt
agent_prompt = "You are a helpful assistant. Write creative product slogans."

def run_agent(task, system_prompt):
    """用当前的 system prompt 执行任务"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{task}")
    ])
    chain = prompt | llm
    return chain.invoke({"task": task}).content

def reflect_and_update(system_prompt, feedback):
    """根据用户反馈来改写 system prompt"""
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI that improves prompts for another AI agent."),
        ("human", "The agent's current system prompt is:\n\n"
                  f"{system_prompt}\n\n"
                  "The user feedback is:\n"
                  f"{feedback}\n\n"
                  "Please propose a new improved system prompt.")
    ])
    chain = reflection_prompt | llm
    return chain.invoke({}).content

# Step 1: agent 生成结果
output1 = run_agent("Promote a new eco-friendly water bottle", agent_prompt)
print("第一次输出:\n", output1)

# Step 2: 用户反馈
feedback = "The slogan is too long. Please be more concise."

# Step 3: agent 反思并更新自己的 system prompt
new_prompt = reflect_and_update(agent_prompt, feedback)
print("\n更新后的 system prompt:\n", new_prompt)

# Step 4: 用更新后的 prompt 再次执行任务
output2 = run_agent("Promote a new eco-friendly water bottle", new_prompt)
print("\n第二次输出:\n", output2)
