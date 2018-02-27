import sys
sys.path.append('/home/brumen/workspace/mrds/')
import config
import numpy as np
import scipy
import scipy.stats

import pricers
import mc
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

import pycuda.curandom
import pycuda.gpuarray as gpa
import pycuda.cumath
import pycuda.reduction
from pycuda.compiler import SourceModule

F1,F2,F3 = 100., 110., 101.
F = [F1,F2,F3]
s1,s2,s3 = 0.25, 0.25, 0.11
s = [s1,s2,s3]
T1,T2,T3 = 1.5,1.7, 2.
T = [T1,T2,T3]
a, w_1, w_2 = 95.25, 2.179, 2.545
K = 28684.
rho_m = np.array([[1., 0.9, 0.25], [0.9, 1., 0.9], [0.25, 0.9, 1.]])
nb_sim = 100000
rns = np.random.multivariate_normal ( np.zeros (3), rho_m, size = nb_sim)
rn1,rn2,rn3 = rns[:,0], rns[:,1], rns[:,2]
rn1_v_d, rn2_v_d, rn3_v_d = gpa.to_gpu (rn1.astype(np.float32)), gpa.to_gpu (rn2.astype(np.float32)), gpa.to_gpu (rn3.astype(np.float32))

F1_sim_d, F2_sim_d, F3_sim_d = mc.mc_one_step_cuda(F1,s1,T1,rn1_v_d), mc.mc_one_step_cuda(F2,s2,T2,rn2_v_d), mc.mc_one_step_cuda(F3,s3,T3,rn3_v_d)

#print np.mean ( np.maximum ( K - F1_sim_d * (a + F1_sim_d * w_2 +  F2_sim_d * w_2 ), 0.))

print pricers.osaka_sim_cuda(F,K, T, s, [a,w_1,w_2], 0.99, [rn1_v_d, rn3_v_d, rn3_v_d],
                             nb_sim)
