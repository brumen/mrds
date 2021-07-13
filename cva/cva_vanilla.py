import datetime
import numpy as np
import pycuda.gpuarray as gpa

from typing          import List

import matplotlib as mpl
mpl.use('TkAgg')

import mrds.cva.cva as cva
import mrds.ds as ds

from mrds.mrds import ComSkew
from mrds.pricers.pricers import black_greeks  # , swap_cva


def cva_eu_call( com       : str
               , mkt_date  : datetime.date
               , K         : float
               , d_expiry  : datetime.date
               , exp_times : List[datetime.date]
               , nb_sim    : int = 50000
               , cp_ind    : str = 'c'
               , dcf       : float = 365.25 ):
    """ CVA exposure for a simple European call option.

    :param com: commodity for which to compute
    :param mkt_date: market date
    :param K: strike price of the option
    :param d_expiry: expiry of the option
    :param exp_times: exposure times.
    :param nb_sim: number of simulations
    :param cp_ind: call, put indicator
    :returns: generator of exposure for the european call option.
    """

    model = ComSkew.from_db(mkt_date, [com] )
    F_sim = model.simulate_curves([com], nb_sim, exp_times, tenor_list=[d_expiry])
    F_sim_by_exp_time = { exp_time: F_sim[com][exp_time_idx, 0, :]
                          for exp_time_idx, exp_time in enumerate(exp_times) }

    vo_obj = model.vol_curve_names(com)

    for exp_time, exp_sims in F_sim_by_exp_time.items():
        ttm = (exp_time - mkt_date).days/dcf

        yield exp_time, black_greeks( exp_sims
                                    , K
                                    , model.DF(ttm)
                                    , vo_obj.implied_vol(d_expiry, exp_sims, ttm )
                                    , ttm
                                    , cp_ind     = cp_ind
                                    , price_only = True
                                    , fast_appx  = True
                                    , )


def swap_compute_exp(F_sims, tenor_t, exp_t, swap_rate, nb_sim, cuda_ind=False):
    """
    compute swap exposure

    :param F_sims: simulation [exp_time, forward_idx, simulation repeat]
    :param tenor_t: tenor times for forwards in F_sims
    :param exp_t: exposure time
    :param swap_rate: swap rate, a number
    """
    nb_exp_points = len(exp_t)
    all_tenors = np.arange(len(tenor_t))
    mpe = np.empty(nb_exp_points)
    mne = np.empty(nb_exp_points)
    me = np.empty(nb_exp_points)

    for exp_one_ind, exp_one_t in enumerate(exp_t):
        tenors_chosen = all_tenors[tenor_t >= exp_one_t]
        if not cuda_ind:
            F_sim_relevant = pricers.swap_cva(F_sims[exp_one_ind, tenors_chosen, :], swap_rate,
                                              cuda_ind=cuda_ind)
            me[exp_one_ind], mpe[exp_one_ind], mne[exp_one_ind] = np.mean(F_sim_relevant), \
                np.mean(F_sim_relevant[F_sim_relevant > 0]), \
                np.mean(F_sim_relevant[F_sim_relevant < 0])

        else:
            if tenors_chosen == []:
                me[exp_one_ind], mpe[exp_one_ind], mne[exp_one_ind] = 0., 0., 0.
            else:
                F_sim_relevant = pricers.swap_cva(F_sims[exp_one_ind][tenors_chosen[0]:(tenors_chosen[-1]+1), :],
                                                  swap_rate,
                                                  cuda_ind=cuda_ind)
                me[exp_one_ind], mpe[exp_one_ind], mne[exp_one_ind] = \
                    np.float32(np.array(gpa.sum(F_sim_relevant).get()))/nb_sim, 1., 1.

    return me, mpe, mne


def cva_swap(mm, params,
             cuda_ind=False):
    """
    cva exposure for swap
    :param mm: market model
    :param params: parameters
    """

    sim_times = params['sim_times']
    sim_times_dt = [ds.convert_str_datetime(st) for st in sim_times]
    sim_times_rev_idx = range(len(sim_times))[::-1]  # reversed index
    nb_sims = params['nb_sims']
    swap_start = params['swap_start']
    swap_start_dt = ds.convert_str_datetime(swap_start)
    swap_end = params['swap_end']
    swap_end_dt = ds.convert_str_datetime(swap_end)
    quantity = params['quantity']

    mm.update_sim_times(sim_times)
    if cuda_ind:
        mm.simulate_curves_cuda(nb_sims)
    else:
        mm.simulate_curves(nb_sims)
    sc = mm.simulated_curves[0]

    def swap_idx(sim_time_dt):
        return [ix for (ix, td) in enumerate(mm.forward_tenors_dt_list[0])
                if (td >= swap_start_dt) and (td <= swap_end_dt) and (td >= sim_time_dt)]

    time_points, nb_tenors, nb_sims = sc.shape
    optimal_value = np.zeros((time_points, nb_sims))
    if cuda_ind:
        sc = sc.get()
    optimal_value[-1, :] = pricers.swap_cva(sc[-1, swap_idx(sim_times_dt[-1]), :],
                                            params['swap_rate'])
    for time_step_idx in sim_times_rev_idx[1:]:
        F_curr = sc[time_step_idx, swap_idx(sim_times_dt[time_step_idx]), :]
        optimal_value[time_step_idx, :] = pricers.swap_cva(F_curr, params['swap_rate'])
    optimal_value *= quantity
    me_v, mpe_v, mne_v, q95_v, q05_v = cva.exposure_compute(optimal_value)
    return {"me": me_v, "mpe": mpe_v, "mne": mne_v, "q95": q95_v, "q05": q05_v}


def main():
    """ examples of functions.
    """

    cva_eu = cva_eu_call( 'WTI'
                        , datetime.date(2015, 4, 1)
                        , 50.
                        , datetime.date(2015, 12, 31)
                        , [datetime.date(2015, 7, 1), datetime.date(2015, 11, 1)]
                        , )

    for e in cva_eu:
        print(e)

main()
