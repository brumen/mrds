import config
import numpy as np
import ctypes
libtp1 = ctypes.CDLL(config.work_dir + 'tp.so')

N1 = 120
a = np.ones((N1,N1,N1,N1)) / np.double(N1**4)
b = 3. * np.ones((N1,N1)) / np.double(N1**2)
c = 2. * np.ones((N1,N1)) / np.double(N1**2)
res = np.zeros((N1,N1))

#this above here is working 
for ind in range(10):
    res = np.zeros((N1,N1))
    libtp1.tensor_prod_1(ctypes.py_object(a),
                         ctypes.py_object(b),
                         ctypes.py_object(c),
                         ctypes.py_object(res))


for ind in range(10):
    res = np.zeros((N1,N1))
    libtp1.tensor_prod_2(ctypes.py_object(a),
                         ctypes.py_object(b),
                         ctypes.py_object(c),
                         ctypes.py_object(res))
