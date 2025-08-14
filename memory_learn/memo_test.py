from mem0 import MemoryClient
client = MemoryClient()

messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! I'll remember your dietary preferences."}
]

result = client.add(messages, user_id="alex")
print(result)
result = client.search("What should I cook for dinner?", user_id="alex")
print(result)
results = client.get_all(user_id="alex")
print(results)



