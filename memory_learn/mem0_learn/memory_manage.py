import time


class MemoryItem:
    def __init__(self, content):
        self.content = content
        self.weight = 1.0  # 初始权重
        self.last_access = time.time()

    def reinforce(self, value=0.5):
        """被访问或确认时强化"""
        self.weight += value
        self.last_access = time.time()

    def decay(self, decay_rate=0.9):
        """随时间衰退"""
        elapsed = time.time() - self.last_access
        # 按天计算衰退
        days_passed = elapsed / 86400
        self.weight *= decay_rate ** days_passed


class MemoryManager:
    def __init__(self):
        self.memories = []

    def add_memory(self, content):
        item = MemoryItem(content)
        self.memories.append(item)
        return item

    def retrieve(self, query, reinforce_value=0.5):
        """检索相关记忆（这里只用简单匹配），访问时强化"""
        results = [m for m in self.memories if query.lower() in m.content.lower()]
        for m in results:
            m.reinforce(reinforce_value)
        return sorted(results, key=lambda x: x.weight, reverse=True)

    def decay_all(self):
        for m in self.memories:
            m.decay()


# ==== 示例使用 ====
manager = MemoryManager()
manager.add_memory("I love sci-fi movies")
manager.add_memory("My favorite food is pizza")
manager.add_memory("I enjoy hiking on weekends")

# 模拟对话访问
print("第一次访问:")
results = manager.retrieve("sci-fi")
for r in results:
    print(r.content, r.weight)

# 模拟一天后衰退
time.sleep(1)  # 用1秒模拟时间过去
manager.decay_all()

print("\n衰退后:")
for m in manager.memories:
    print(m.content, round(m.weight, 2))

# 再次访问强化
results = manager.retrieve("sci-fi")
print("\n再次访问强化后:")
for r in results:
    print(r.content, round(r.weight, 2))
