from calendar import c
import functools
import re
import time

def timer(func):
    '''这是一个定时器装饰器的实现'''
    @functools.wraps(func) # 保留原函数的元信息
    def wrapper(*args, **kwargs):
        start_time=time.perf_counter()
        result=func(*args,**kwargs)
        end_time=time.perf_counter()
        print(f"函数{func.__name__}执行时间：{end_time-start_time}秒")
        return result
    return wrapper

def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"第{attempt + 1}/{max_attempts}次尝试失败: {e}")
            raise Exception(f"达到最大尝试{max_attempts}次数")
        return wrapper
    return decorator

def main():
    pass

@timer
def process_data():
    time.sleep(1)
    return "数据处理完成"

@timer
@retry(max_attempts=5)
def unstable_network_call():
    import random
    if random.random() < 0.01:
        raise Exception("模拟网络调用失败")
    return "网络调用成功"

def cache(func):
    _cache={}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args not in _cache:
            _cache[args]=func(*args, **kwargs)
        return _cache[args]
    return wrapper
@timer
@cache
def llm_call(n):
    time.sleep(n)
    return n*2
if __name__ == "__main__":
    # main()
    # res=process_data()
    # print(res)
    # unstable_network_call()
    print(llm_call(1))
    print(llm_call(2))
    print(llm_call(3))
    print(llm_call(1))
    print(llm_call(2))
    print(llm_call(3))
