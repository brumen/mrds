import config
import opd_avx
import numpy as np

N = 128000
# N = 11
r = np.random.rand(N)
a = np.random.rand(N)
b = np.random.rand(N)
c = np.random.rand(N)
d = np.random.rand(N)
y = np.empty(N)


def t1():
    opd_avx.add4(r, a, b, c, d, y, N)
    # return y

def t2():
    y = r - (a + b + c + d)
    # return y

def t3(N1):
    r = np.random.rand(N1)
    a = np.random.rand(N1)
    b = np.random.rand(N1)
    c = np.random.rand(N1)
    d = np.random.rand(N1)
    y = np.empty(N1)
    opd_avx.add4(r, a, b, c, d, y, N1)
    y2 = r - (a + b + c + d)
    return y- y2

