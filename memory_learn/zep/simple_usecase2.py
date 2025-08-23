import os
import uuid
from dotenv import load_dotenv
from zep_cloud.client import Zep
from zep_cloud import Message

# ======================
# 1. 初始化 Zep 客户端
# ======================
load_dotenv(override=True)
ZEP_API_KEY = os.getenv("ZEP_API_KEY")
client = Zep(api_key=ZEP_API_KEY)

# ======================
# 2. 用户管理
# ======================
def add_user(user_id, email, first_name, last_name):
    """添加一个新用户"""
    user = client.user.add(
        user_id=user_id,
        email=email,
        first_name=first_name,
        last_name=last_name,
    )
    print(f"✅ 用户 {user_id} 已创建:", user)
    return user

def get_user(user_id):
    """获取用户信息"""
    user = client.user.get(user_id)
    print("ℹ️ 用户信息:", user)
    return user

# ======================
# 3. 会话管理（Thread）
# ======================
def create_thread(user_id):
    """为用户创建新的对话线程"""
    thread_id = uuid.uuid4().hex
    client.thread.create(thread_id=thread_id, user_id=user_id)
    print("🆕 新建线程:", thread_id)
    return thread_id

# ======================
# 4. 添加记忆
# ======================
def add_memory(user_id, bot_id, thread_id, user_message, bot_message):
    """将用户和 AI 的对话存入记忆"""
    messages = [
        Message(name=user_id, role="user", content=user_message),
        Message(name=bot_id, role="assistant", content=bot_message),
    ]
    client.thread.add_messages(thread_id, messages=messages)
    print(f"💾 已存入记忆: {user_message} -> {bot_message}")

# ======================
# 5. 获取长期记忆摘要
# ======================
def get_user_context(thread_id, mode="summary"):
    """获取用户上下文（支持 summary/basic 两种模式）"""
    memory = client.thread.get_user_context(thread_id=thread_id, mode=mode)
    print(f"📌 用户上下文（{mode}）:\n", memory.context)
    return memory.context


def search_in_graph(user_id, query_text, center_node_uuid=None):
    """
    在 Zep 知识图谱中搜索与 query_text 相关的节点或事实
    """
    client = Zep(api_key=ZEP_API_KEY)
    if center_node_uuid:
        results = client.graph.search(
            user_id=user_id,
            query=query_text,
            reranker="node_distance",
            center_node_uuid=center_node_uuid  # 可选，指定搜索中心
        )
    else:
        results = client.graph.search(
            user_id=user_id,
            query=query_text
        )

    relevant_nodes = results.nodes
    relevant_edges = results.edges

    print("=== 节点 ===")
    if relevant_nodes:
        for node in relevant_nodes:
            print(node)

    print("=== 相关事实 ===")
    if relevant_edges:
        for edge in relevant_edges:
            print(edge.fact)


def auto_graph_search(user_id, query_text):
    client = Zep(api_key=ZEP_API_KEY)

    # 先尝试列出用户所有节点
    nodes = client.graph.node.get_by_user_id(user_id=user_id)

    # 简单关键词匹配（可以换成更智能的 NER 模型）
    center_node_uuid = None
    for node in nodes:
        if query_text in node.labels:
            center_node_uuid = node.uuid
            break

    # 如果找到中心节点，就指定搜索，否则全局搜索
    return search_in_graph(user_id, query_text, center_node_uuid)



# ======================
# 6. 演示流程
# ======================
if __name__ == "__main__":
    # 定义两个用户：human user 和 bot
    user_id = "harry_user1"
    bot_id = "ai_bot1"

    # 创建用户
    # add_user(user_id, "harry@example.com", "Harry", "Liu")
    # add_user(bot_id, "bot@example.com", "AI", "Assistant")
    # get_user(user_id)
    # get_user(bot_id)
    #
    # # 创建对话线程（以 human user 为主）
    # thread_id = create_thread(user_id)
    #
    # # 模拟一次对话
    # user_msg = "hi, I am harry, I love eating noodles."
    # bot_msg = "Hi Harry! 🍜 Nice to meet you. What kind of noodles do you like the most—ramen, pasta, stir-fry, or something else?"
    # add_memory(user_id, bot_id, thread_id, user_msg, bot_msg)
    #
    # # 再来一次对话
    # user_msg2 = "noodles is not favourate any more . I think rice is better now"
    # bot_msg2 = "Got it—so rice has taken the crown! 🍚 Do you prefer it plain, fried, or with curry/sauce?"
    # add_memory(user_id, bot_id, thread_id, user_msg2, bot_msg2)

    thread_id = "af9b84ce69064f7ea4d83014c249cb15"

    # 获取用户记忆摘要
    # get_user_context(thread_id, "summary")
    get_user_context(thread_id, "basic")
    # search_in_graph(user_id, "what does harry like?")
    # auto_graph_search(user_id, "Harry")
