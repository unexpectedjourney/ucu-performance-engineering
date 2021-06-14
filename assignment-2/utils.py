import time

def timeit(method):
    def timed(*args, **kw):
        start_point = time.time()
        result = method(*args, **kw)
        end_point = time.time()
        print(f"{method.__name__}: {end_point - start_point}")
        return result
    return timed
