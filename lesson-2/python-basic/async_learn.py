import asyncio
import re
from reprlib import aRepr
import time
from tkinter import wantobjects
from turtle import resetscreen

async def worker(name, delay):
    print(f"Worker {name} started")
    await asyncio.sleep(delay)
    print(f"Worker {name} finished")
    return f"Worker {name} result"

async def main():
    result = await asyncio.gather(worker("A", 10), worker("B", 2), worker("C", 3))
    return result

async def task_main():
    a_task = asyncio.create_task(worker("A", 10))
    b_task = asyncio.create_task(worker("B", 2))
    c_task = asyncio.create_task(worker("C", 3))
    res_a = await a_task
    res_b = await b_task
    res_c = await c_task
    return [res_a, res_b, res_c]

if __name__ == "__main__":
    print(f"Starting:{time.strftime('%Y-%m-%d %H:%M:%S')}")
    # result = asyncio.run(main())
    result = asyncio.run(task_main())
    print(f"Result: {result}")
    print(f"En:{time.strftime('%Y-%m-%d %H:%M:%S')}")
