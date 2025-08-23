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


def get_user_context(thread_id, mode = "summary"):
    client = Zep(api_key=ZEP_API_KEY)
    # Get memory for the thread
    memory = client.thread.get_user_context(thread_id=thread_id, mode= mode)

    # Access the context block (for use in prompts)
    context_block = memory.context
    print(context_block)
    return context_block


if __name__ == '__main__':
    thread_id = "0be0c6daf6fe49f79406b19925a22ef1"
    user_context = get_user_context(thread_id)
    user_context = get_user_context(thread_id, "basic")
