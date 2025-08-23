
from dotenv import load_dotenv
import os

# 加载 .env 文件中的变量
load_dotenv(override=True)

# 通过 os.getenv 获取
ZEP_API_KEY = os.getenv("ZEP_API_KEY")

print(ZEP_API_KEY)

from zep_cloud.client import Zep


def add_user(user_id):
    client = Zep(api_key=ZEP_API_KEY)

    new_user = client.user.add(
        user_id=user_id,
        email="user@example.com",
        first_name="Harry",
        last_name="Liu",
    )
    print(new_user)

def get_user(user_id):
    client = Zep(api_key=ZEP_API_KEY)
    user = client.user.get(user_id)
    print(user)
    return user

def update_user(user_id):

    client = Zep(api_key=ZEP_API_KEY)
    updated_user = client.user.update(
        user_id=user_id,
        email="updated_user@example.com",
        first_name="Harry",
        last_name="he",
    )
    print(updated_user)

def delete_user(user_id):
    client = Zep(api_key=ZEP_API_KEY)
    client.user.delete(user_id)

def get_threads(user_id):
    client = Zep(api_key=ZEP_API_KEY)
    threads = client.user.get_threads(user_id)

    for thread in threads:
        print(thread)

def list_users():
    client = Zep(api_key=ZEP_API_KEY)
    result = client.user.list_ordered(page_size=10, page_number=1)
    print(result)


def get_user_node(user_id):
    client = Zep(api_key=ZEP_API_KEY)
    results = client.user.get_node(user_id=user_id)
    user_node = results.node
    print(user_node.summary)

def find_node_by_graph(user_id):
    client = Zep(api_key=ZEP_API_KEY)
    # Initialize the Zep client

    nodes = client.graph.node.get_by_user_id(user_id=user_id)
    for node in nodes:
        print(node)

def search_in_graph(user_id, query):
    client = Zep(api_key=ZEP_API_KEY)
    results = client.graph.search(
        user_id=user_id,
        query=query,  # To help narrow down the nodes you have to manually search
        scope="nodes",
        min_score=1
    )
    relevant_nodes = results.nodes

    for relevant_node in relevant_nodes:
        print(relevant_node)

def get_relevant_fact(user_id, query, center_node_uuid):
    client = Zep(api_key=ZEP_API_KEY)
    results = client.graph.search(
        user_id=user_id,
        query=query,
        reranker="node_distance",
        center_node_uuid=center_node_uuid,
    )
    relevant_edges = results.edges
    relevant_facts = [edge.fact for edge in relevant_edges]

    for relevant_fact in relevant_facts:
        print(relevant_fact)


def get_facts(user_id, center_node_uuid):
    client = Zep(api_key=ZEP_API_KEY)
    edges = client.graph.edge.get_by_user_id(user_id=user_id)
    connected_edges = [edge for edge in edges if
                       edge.source_node_uuid == center_node_uuid or edge.target_node_uuid == center_node_uuid]
    relevant_facts = [edge.fact for edge in connected_edges]
    for relevant_fact in relevant_facts:
        print(relevant_fact)


if __name__ == '__main__':
    # user_id = "hl77319"
    # user_id = "user_123"
    user_id = "ai bot"
    add_user(user_id)
    # update_user(user_id)
    # delete_user(user_id)
    get_user(user_id)
    # get_threads(user_id)
    # list_users()
    # get_user_node(user_id)
    # find_node_by_graph(user_id)
    # search_in_graph(user_id, "harry")
    # get_facts(user_id, "d42ae90c-ecd2-4940-9377-f98c22ffe985")
    # get_relevant_fact(user_id,"harry", "d42ae90c-ecd2-4940-9377-f98c22ffe985" )
