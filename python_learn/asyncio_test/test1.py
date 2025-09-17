import asyncio


async def foo(name, delay):
    print(f"{name} start")
    await asyncio.sleep(delay)
    print(f"{name} end")
    return name


async def main():
    task1 = asyncio.create_task(foo("A", 2))
    task2 = asyncio.create_task(foo("B", 2))

    print("任务已创建，可以先做点别的事...")

    # 等待单个任务
    result1 = await task1
    result2 = await task2
    print("结果:", result1, result2)


asyncio.run(main())