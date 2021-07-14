""" Vanilla CVA computations.
"""

import datetime

from typing          import List

from mrds.mrds            import ComSkew
from mrds.pricers.pricers import black_greeks
from mrds.instruments.vanilla_swap import VanillaSwap
from mrds.forward_curve import FwdCurve


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


def cva_swap( swap : VanillaSwap
            , mkt_date: datetime.date
            , exp_times: List[datetime.date]
            , nb_sim: int = 50000
            , dcf: float = 365.25 ):
    """ CVA exposure for swap
    """

    com = swap.index_name
    payments = swap.payments()

    model = ComSkew.from_db(mkt_date, [com] )
    F_sim = model.simulate_curves([com], nb_sim, exp_times, tenor_list=payments)
    F_sim_by_exp_time = { exp_time: F_sim[com][exp_time_idx, :, :]  # Index 1 are payment blocks
                          for exp_time_idx, exp_time in enumerate(exp_times) }

    for exp_date, exp_sims in F_sim_by_exp_time.items():
        swap.pricing_date = exp_date

        swap_sims_for_exp = []
        for sim_nb in range(nb_sim):
            swap.index_curve = FwdCurve( mkt_date
                                       , com
                                       , payments
                                       , F_sim_by_exp_time[exp_date][:, sim_nb]
                                       , dcf = dcf )
            swap_sims_for_exp.append(swap.PV())

        yield exp_date, swap_sims_for_exp


def main_eu_call(mkt_date, sample_exp_times):
    """ EU CALL EXAMPLE
    """

    cva_eu = cva_eu_call( 'WTI'
                        , mkt_date
                        , 50.
                        , datetime.date(2015, 12, 31)
                        , sample_exp_times
                        , )

    exp_eu = list(cva_eu)
    return exp_eu


def main_swap(mkt_date, sample_exp_times):
    """ CVA SWAP EXAMPLE
    """

    cva_swap_2 = cva_swap( VanillaSwap(mkt_date, mkt_date, maturity='1Y'), mkt_date, sample_exp_times)

    exp_swap = list(cva_swap_2)
    return exp_swap


def main():
    MKT_DATE = datetime.date(2015, 4, 1)
    sample_exp_times = [datetime.date(2015, 7, 1), datetime.date(2015, 11, 1)]

    res_swap = main_swap(MKT_DATE, sample_exp_times)


main()
