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

import pycuda.gpuarray as gpa

from multiprocessing import Pool
import time

def osaka_run (params):
    T_sim_idx, F_sim, T, T_l, df, K, s, rho, aw, Khat, lam1, nb_sim, nb_sim2 = params

    osaka_price_local = np.zeros ((nb_sim2, 2))

    T_sim= T_l[T_sim_idx]
    for sim_nb in np.arange(nb_sim2):
        F_local = F_sim[T_sim_idx, sim_nb, :]
        T_local = T - T_sim

        osaka_price_local[sim_nb, :] = pricers.osaka_sim (F_local, K, T_local, s, rho,
                                                          aw, df, nb_sim, lam1, Khat)[0:2]

    return osaka_price_local


def osaka_run_nocuda (params):
    T_sim_idx, F_sim, T, T_l, df, K, s, rho, aw, Khat, lam1, nb_sim, nb_sim2 = params

    osaka_price_local = np.zeros ((nb_sim2, 2))

    r_1, r_2, r_3 = rho
    rho_m = np.array([[1., r_1, r_2 ], [r_1, 1., r_3], [r_2, r_3, 1.]])

    rns = np.random.multivariate_normal (np.zeros(3), rho_m, size = nb_sim)

    T_sim= T_l[T_sim_idx]
    for sim_nb in np.arange(nb_sim2):
        F_local = F_sim[T_sim_idx, sim_nb, :]
        T_local = T - T_sim

        osaka_price_local[sim_nb, :] = pricers.osaka_sim_nocuda (F_local, K, T_local, s,
                                                                 aw, df, rns, nb_sim, lam1, Khat)

    return osaka_price_local



def osaka_run_cuda (params):
    T_sim_idx, F_sim, T, T_l, df, K, s, rho, aw, Khat, lam1, nb_sim, nb_sim2 = params

    osaka_price_local = np.zeros (nb_sim2)

    r_1, r_2, r_3 = rho
    rho_m = np.array([[1., r_1, r_2 ], [r_1, 1., r_3], [r_2, r_3, 1.]])

    rns = np.random.multivariate_normal ( np.zeros (3), rho_m, size = nb_sim)
    Fsim_h = np.zeros ( (nb_sim,3)) # to transfer to device 
    Fsim_d = [ gpa.to_gpu (Fsim_h[:,0].astype(np.float32)), 
               gpa.to_gpu (Fsim_h[:,1].astype(np.float32)), 
               gpa.to_gpu (Fsim_h[:,2].astype(np.float32)) ]
    rn1,rn2,rn3 = rns[:,0], rns[:,1], rns[:,2]
    rn1_v_d, rn2_v_d, rn3_v_d = gpa.to_gpu (rn1.astype(np.float32)), gpa.to_gpu (rn2.astype(np.float32)), gpa.to_gpu (rn3.astype(np.float32))
    rn_coll = [rn1_v_d, rn2_v_d, rn3_v_d]

    T_sim= T_l[T_sim_idx]
    for sim_nb in np.arange(nb_sim2):
        F_local = F_sim[T_sim_idx, sim_nb,:]
        T_local = T - T_sim

        #print "qw = ", pricers.osaka_sim_cuda (F_local, K, T_local, s, aw, df, rn_coll, nb_sim)
        osaka_price_local[sim_nb] = pricers.osaka_sim_cuda (F_local, Fsim_d, K, T_local, s, aw, df, rn_coll, nb_sim)


    return osaka_price_local


# main program starts 

F = np.array([115., 110., 101.])
K = 28684.
T_end = 2.
nb_sim = 1000000
nb_sim2 = 100
gpu_ind = True
T = np.array ([T_end - 9./12., T_end - 6./12., T_end - 5./12.])
s = np.array([0.25, 0.25, 0.12])
rho = np.array([0.9, 0.25, 0.25])
r_1, r_2, r_3 = rho 
rho_m = np.array([[1., r_1, r_2 ], [r_1, 1., r_3], [r_2, r_3, 1.]])
df = 0.99
aw = np.array([95.25, 2.179, 2.545])
Khat = (K/F[2] - aw[0]) / (aw[1] + aw[2])
lam1 = 1.
T_l = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5])
F_sim_l = mc.mc_mult_steps(F, s, T_l, rho_m, nb_sim2)

#p = Pool(4)
T_l_len = len (T_l)
params = zip (np.arange(T_l_len), [F_sim_l] * T_l_len, [T] * T_l_len, 
              [T_l] * T_l_len, [df] * T_l_len, [K] * T_l_len, [s] * T_l_len, 
              [rho] * T_l_len, [aw] * T_l_len, [Khat] * T_l_len, 
              [lam1] * T_l_len, [nb_sim] * T_l_len, [nb_sim2] * T_l_len )


t1 = time.time()
if gpu_ind == True:
    osaka_sim_l = map (osaka_run_cuda, params)
else:
    osaka_sim_l = map (osaka_run_nocuda, params)
t2 = time.time() - t1

print "tdiff = ", t2

# NEW 
MRD_mean = [np.mean (x) for x in osaka_sim_l ]
MRD_q95 = [ scipy.stats.mstats.mquantiles (x, prob = 0.95) for x in osaka_sim_l]
MRD_q05 = [ scipy.stats.mstats.mquantiles (x, prob = 0.05) for x in osaka_sim_l]

res = np.array ([ MRD_mean, MRD_q95, MRD_q05 ])

np.savetxt ("osaka_out1.csv", res, delimiter = ',')

print "Finished = ", res

p1, = plt.plot(T_l, MRD_mean, label = 'mrd mean')
p2, = plt.plot(T_l, MRD_q95, label = 'mrd q95')
p3, = plt.plot(T_l, MRD_q05, label = 'mrd q05')

plt.legend ([p1,p2,p3], ['mrd mean', 'mrd q95', 'mrd q05'])

plt.show()







# --- OLD, 

# CMG_mean = [np.mean ( x[:,1]) for x in osaka_sim_l ]
# MRD_mean = [np.mean ( x[:,0]) for x in osaka_sim_l ]
# CMG_q95 = [ scipy.stats.mstats.mquantiles (x[:,1], prob = 0.95) for x in osaka_sim_l]
# MRD_q95 = [ scipy.stats.mstats.mquantiles (x[:,0], prob = 0.95) for x in osaka_sim_l]
# CMG_q05 = [ scipy.stats.mstats.mquantiles (x[:,1], prob = 0.05) for x in osaka_sim_l]
# MRD_q05 = [ scipy.stats.mstats.mquantiles (x[:,0], prob = 0.05) for x in osaka_sim_l]

# res = np.array ([ CMG_mean, MRD_mean, CMG_q95, MRD_q95, CMG_q05, MRD_q05 ])

# np.savetxt ("osaka_out1.csv", res, delimiter = ',')

# print "Finished = ", res

# p1, = plt.plot(T_l, CMG_mean, label = 'cmg mean')
# p2, = plt.plot(T_l, MRD_mean, label = 'mrd mean')
# p3, = plt.plot(T_l, CMG_q95, label = 'cmg q95')
# p4, = plt.plot(T_l, MRD_q95, label = 'cmg q95')
# p5, = plt.plot(T_l, CMG_q05, label = 'cmg q05')
# p6, = plt.plot(T_l, MRD_q05, label = 'cmg q05')

# plt.legend ([p1,p2,p3,p4,p5,p6], ['cmg mean','mrd mean', 'cmg q95', 'mrd q95', 
#                                   'cmg q05', 'mrd q05'])

# plt.show()
