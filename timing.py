from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        msg = f'{func.__name__} took {end - start} seconds'
        logger.debug(msg)
        return result
    return wrapper