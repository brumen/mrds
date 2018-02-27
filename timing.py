#
# timing functions 
import config 
import time


def time_cuda_call(start_event, end_event, f, *args, **kwargs):
    """
    stop-times the cuda call
    """
    start_event.record()
    result = f(*args, **kwargs)
    end_event.record()
    end_event.synchronize()
    t = start_event.time_till(end_event) * 1e-3
    return t, result


def time_normal_call(f, *args, **kwargs):
    """
    stop-time a normal call
    """
    t = time.time()
    res = f(*args, **kwargs)
    t = time.time() - t
    return t, res
