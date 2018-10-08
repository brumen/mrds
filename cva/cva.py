import numpy as np
import scipy
import scipy.stats 

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def exposure_compute(p_mat):
    me_v  = np.mean(p_mat, axis=1)  # average by rows
    mpe_v = np.mean(p_mat * (p_mat >= 0.), axis=1)
    mne_v = np.mean(p_mat * (p_mat <= 0.), axis=1)
    q95_v = np.array([scipy.stats.mstats.mquantiles(x, prob=0.95) for x in p_mat]).ravel()
    q05_v = np.array([scipy.stats.mstats.mquantiles(x, prob=0.05) for x in p_mat]).ravel()
    return {"me": me_v, "mpe": mpe_v, "mne": mne_v, "q95": q95_v, "q05": q05_v}


def exposure_display(res,
                     exposure_days=None,
                     legend_location='upper left'):
    me, mpe, mne, q95, q05 = res['me'], res['mpe'], res['mne'], res['q95'], res['q05']
    sim_times_plot = np.arange(len(me)) if exposure_days is None else exposure_days

    p1, = plt.plot(sim_times_plot, me, label='me')
    p2, = plt.plot(sim_times_plot, mpe, label='mpe')
    p3, = plt.plot(sim_times_plot, mne, label='mne')
    p4, = plt.plot(sim_times_plot, q95, label='q95')
    p5, = plt.plot(sim_times_plot, q05, label='q05')
    plt.legend([p1, p2, p3, p4, p5],
               ['me', 'mpe', 'mne', 'q95', 'q05'],
               loc=legend_location)
    plt.show()
