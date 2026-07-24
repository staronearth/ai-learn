import asyncio

sem=asyncio.Semaphore(5)

async def worker(name,delay):
    async with sem:
        print(f"Worker {name} started")
        await asyncio.sleep(delay)
        print(f"Worker {name} finished")

async def main():
    tasks = [worker(f"w{i}", i) for i in range(10)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
