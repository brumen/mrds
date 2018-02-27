import sys
sys.path.append('/home/brumen/workspace/mrds/')
import config
import numpy as np

import mc


F = np.array([100.])
K = 100.
T_end = 2.
nb_sim = 10000

s = np.array([0.25])  # vol
rho_m = np.array([[1.]])  # rho, not relevant for 1 future
df = 0.995  # discount factor 5% yearly discount rate
M = 100
T_l = np.linspace(0.1, 10., M)  # number of simulation steps
F_sim_l = mc.mc_mult_steps(F, s, T_l, rho_m, nb_sim)[:, :, 0]
print np.average(F_sim_l[-1, :])
reg = 9  # number of basis functions


def put_payoff(F):
    return np.maximum(K - F, 0.)


def lsmm(F_sim_l):
    h = put_payoff(F_sim_l)  # exercise value
    V = put_payoff(F_sim_l)  # continuation value

    for t in range(M-2, -1, -1):
        rg = np.polyfit(F_sim_l[t, :], V[t+1, :]*df, reg)
        C = np.polyval(rg, F_sim_l[t, :])
        V[t, :] = np.where(h[t, :] > C, h[t, :], V[t+1, :] * df)

    return np.sum(V[0, :])/nb_sim

print lsmm(F_sim_l)
