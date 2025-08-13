import asyncio
import threading
import time

# 异步协程版本
async def async_task(name):
    print(f"[协程] {name} 开始")
    await asyncio.sleep(2)  # 模拟 I/O
    print(f"[协程] {name} 结束")

# 线程版本
def thread_task(name):
    print(f"[线程] {name} 开始")
    time.sleep(2)  # 模拟 I/O
    print(f"[线程] {name} 结束")

async def main():
    # 协程并发运行
    await asyncio.gather(
        async_task("任务1"),
        async_task("任务2"),
        async_task("任务3"),
    )
    print("aa")

if __name__ == "__main__":
    print("=== 协程版本 ===")
    asyncio.run(main())  # 这里 main() 是一个协程对象

    print("\n=== 线程版本 ===")
    threads = []
    for i in range(3):
        t = threading.Thread(target=thread_task, args=(f"任务{i+1}",))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
