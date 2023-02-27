import os
from time import perf_counter
from functools import wraps
import psutil
from psutil._common import bytes2human


def timeit(func):
    """ A decorator for function processes time """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def mem_profile(func):
    """ A decorator for memory profiling """

    def process_memory():
        # inner psutil function, only to be used in this context
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    @wraps(func)
    def mem_profile_wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        usage_string = bytes2human(mem_after - mem_before)
        print(f'Function {func.__name__}{args} {kwargs} consumed memory: {usage_string}')
        return result
    return mem_profile_wrapper