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


def add_memory(user_id, thread_id,user_message, bot_message):
    client = Zep(api_key=ZEP_API_KEY)

    messages = [
        Message(
            name=user_id,
            role="user",
            content=user_message,
        ),
        Message(
            name="ai bot",
            role="assistant",
            content=bot_message,
        )

    ]
    episode_uuids = client.thread.add_messages(thread_id, messages=messages)
    print("episode_uuids", episode_uuids)


if __name__ == '__main__':
    user_id = "hl77319"
    # user_id = "user_123"

    # thread_id = create_thread()
    thread_id = "0be0c6daf6fe49f79406b19925a22ef1"
    # user_content = "I am harry, how are you"
    # bot_content = "Hi Harry! I’m doing well, thanks for asking 😊 How are you doing today?"
#     user_content = "I like watching movie and singing song"
#     bot_content = """That’s awesome, Harry! 🎬🎶
# What kind of movies do you usually enjoy—action, comedy, sci-fi, romance? And do you like singing along to your favorite songs, or just listening?
# """
    user_content = "I want to learn python program, please teach me some basic code in python"
    bot_content = """
    That’s a great choice, Harry! 🐍 Python is one of the easiest and most powerful languages to start with. Let’s go step by step.

Here are some **basic Python concepts with examples**:

---

### 1. **Printing something**

```python
print("Hello, Harry!")
```

👉 This will show:

```
Hello, Harry!
```

---

### 2. **Variables (storing values)**

```python
name = "Harry"
age = 20
print("My name is", name, "and I am", age, "years old.")
```

---

### 3. **Taking input from the user**

```python
name = input("Enter your name: ")
print("Hello", name)
```

---

### 4. **If-else (decision making)**

```python
age = int(input("Enter your age: "))

if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")
```

---

### 5. **Loop (repeating code)**

```python
for i in range(5):
    print("This is line", i+1)
```

---

✨ Your first small program:
Let’s write a program that asks for your favorite movie and then prints it back:

```python
movie = input("What is your favorite movie? ")
print("Wow! I also like", movie)
```

---

Would you like me to prepare a **step-by-step learning plan** for Python (like what to learn first, second, third…), or do you want me to just give you fun little codes to try?

    """

    add_memory(user_id, thread_id, user_content, bot_content)

    get_messages(thread_id)
