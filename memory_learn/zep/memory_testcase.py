import uuid

from dotenv import load_dotenv
import os

from zep_cloud import Message

# 加载 .env 文件中的变量
load_dotenv(override=True)

# 通过 os.getenv 获取
ZEP_API_KEY = os.getenv("ZEP_API_KEY")

print(ZEP_API_KEY)

from zep_cloud.client import Zep

def create_thread():
    client = Zep(api_key=ZEP_API_KEY)

    thread_id = uuid.uuid4().hex  # A new thread identifier

    client.thread.create(
        thread_id=thread_id,
        user_id=user_id,
    )
    print("thread_id", thread_id)
    return thread_id

def get_messages(thread_id):
    client = Zep(api_key=ZEP_API_KEY)
    messages = client.thread.get(thread_id)
    for message in messages:
        print(message)
    return messages

def delete_thread(thread_id):
    client = Zep(api_key=ZEP_API_KEY)
    client.thread.delete(thread_id)

def list_all_thread():
    client = Zep(api_key=ZEP_API_KEY)
    # List the first 10 Threads
    result = client.thread.list_all(page_size=10, page_number=1)
    for thread in result.threads:
        print(thread)


def add_memory(user_id, thread_id,message_content):
    client = Zep(api_key=ZEP_API_KEY)

    messages = [
        Message(
            name=user_id,
            role="user",
            content=message_content,
        )
    ]
    episode_uuids = client.thread.add_messages(thread_id, messages=messages)
    print("episode_uuids", episode_uuids)


if __name__ == '__main__':
    user_id = "hl77319"
    # user_id = "user_123"

    # thread_id = create_thread()
    thread_id = "0be0c6daf6fe49f79406b19925a22ef1"
    message_content = "I am harry, how are you"
    add_memory(user_id, thread_id, message_content)

    get_messages(thread_id)
