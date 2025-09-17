import asyncio
import httpx
import time

URL_SYNC = "http://127.0.0.1:8000/sync"
URL_ASYNC = "http://127.0.0.1:8000/async"

async def worker(client, url):
    try:
        resp = await client.get(url, timeout=10.0)
        return resp.status_code
    except Exception as e:
        return str(e)

async def run_load_test(url, total_requests=200, concurrency=50):
    print(f"\n>>> 压测 {url} | 总请求数={total_requests}, 并发={concurrency}")
    start = time.time()

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(total_requests):
            tasks.append(worker(client, url))
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    success = sum(1 for r in results if r == 200)
    errors = [r for r in results if r != 200]

    print(f"耗时: {elapsed:.2f} 秒")
    print(f"成功请求: {success}, 失败请求: {len(errors)}")
    if errors:
        print("部分错误示例:", errors[:5])

if __name__ == "__main__":
    asyncio.run(run_load_test(URL_SYNC, total_requests=300, concurrency=50))
    asyncio.run(run_load_test(URL_ASYNC, total_requests=300, concurrency=50))
