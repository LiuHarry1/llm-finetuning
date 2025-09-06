import redis
import time
import threading

# Redis 连接
redis_conn = redis.Redis(host="localhost", port=6379, db=0)

# 全局参数
QPS_LIMIT = 60          # 最大每秒请求数
WINDOW = 1              # 时间窗口，单位秒
KEY = "llm_rate_limit"  # Redis 键

def acquire_token():
    """
    使用 Redis 的 zset 实现滑动窗口限流
    返回 True 表示可以请求 LLM，False 表示需要等待
    """
    now = int(time.time() * 1000)  # 毫秒
    window_start = now - WINDOW * 1000

    pipeline = redis_conn.pipeline()
    pipeline.zremrangebyscore(KEY, 0, window_start)  # 删除过期的记录
    pipeline.zcard(KEY)                                # 当前窗口内请求数
    pipeline.zadd(KEY, {str(now): now})                # 添加当前请求
    pipeline.expire(KEY, WINDOW + 1)
    _, count, _, _ = pipeline.execute()
    # print(redis_conn.keys("*"))
    if count <= QPS_LIMIT:
        return True
    else:
        return False

def worker_task(task_id):
    """
    模拟 LLM 调用任务
    """
    while not acquire_token():
        time.sleep(0.01)  # 等待可用 token

    print(f"[{time.time():.3f}] Task {task_id} -> executed")

if __name__ == "__main__":
    # 模拟 100 个并发请求
    threads = []
    for i in range(100):
        t = threading.Thread(target=worker_task, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All tasks completed")
