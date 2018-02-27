# Unit Testing script of cva module
# 
import config
import numpy as np
import datetime as dt
import ds
import vols
import cva
import mrds
import test_params
import cva_tolling
import cva_vanilla
import cva_rt


def test_cva_swap(cuda_ind=False):
    params = dict()
    params['nb_sims'] = 5000
    params['today'] = '20150401'
    params['exp_days'] = np.linspace(3, 365, 50)# [3, 7, 14, 21, 31, 62, 184, 250, 365]
    params['today_dt'] = ds.convert_str_datetime(params['today'])
    params['sim_times_dt'] = [params['today_dt'] + dt.timedelta(day)
                              for day in params['exp_days']]
    params['sim_times'] = [ds.convert_datetime_str(d) for d in params['sim_times_dt']]
    params['swap_start'] = '20150410'
    params['swap_end'] = '20151231'
    params['swap_rate'] = 55.
    params['quantity'] = 100.
    mm = mrds.mrds_calib('WTI', params['today'], 10)
    # mm.model_skew_ln_ind = 'ln_ln'
    on = cva.cva_swap(mm, params, cuda_ind=cuda_ind)
    cva.exposure_display(on, exposure_days=params['exp_days'],
                         legend_location='upper right')
    return on


def test_path_dep_opt():
    print cva.cva_path_dep_opt_2('WTI', '20150401',
                                 ['20150501', '20150601'],
                                 ['20150801', '20151231'], 5,
                                 50., 40., 0.95)


def test_eu_call(debug_ind=False,
                 model='ln_ln',
                 market_date='20140401'):
    exp_times = [1, 7, 14, 21, 30, 60, 180]
    exp_times_str = [ds.convert_datetime_str(ds.convert_str_datetime(market_date) +
                                             dt.timedelta(days=d))
                     for d in exp_times]

    return cva_vanilla.cva_eu_call('WTI', market_date, 65., '20150601',
                           exp_times_str, model_skew_ln_ind=model)


def test_tolling_1(cuda_ind=False):
    """
    CVA exposure profile for a tolling deal
    """
    params = test_params.test_simplest_toll_params()
    params.toll_start_date = '20150501'
    params.toll_end_date = '20161231'
    power_mkt = test_params.test_simplest_toll_market()
    market_date = '20150401'
    exposure_day_diff = [1, 3, 5, 10, 30, 60, 180]
    market_date_dt = ds.convert_str_datetime(market_date)
    power_mkt['fwd_date'] = market_date
    exposure_dates = [ds.convert_datetime_str(market_date_dt + dt.timedelta(days=d))
                      for d in exposure_day_diff]
    power_mkt['nb_sim'] = 5000
    exposure = cva_tolling.cva_tolling(power_mkt, power_mkt['fwd_date'],
                                       params.toll_start_date,
                                       params.toll_end_date,
                                       exposure_dates,
                                       params, 50000, 50000,
                                       parallel_ind=False,
                                       cuda_ind=cuda_ind, 
                                       pricing_model_ind='skew')
    print "Exposure =", exposure
    return True


def test_tolling_2(cuda_ind=False):
    """
    CVA exposure profile for a tolling deal VERY LONG DATED Toll
    """
    params = test_params.test_simplest_toll_params()
    params.toll_start_date = '20150501'
    params.toll_end_date = '20261031'
    power_mkt = test_params.test_simplest_toll_market()
    market_date = '20150401'
    exposure_day_diff = [1, 3, 5, 10, 30, 60, 180]
    market_date_dt = ds.convert_str_datetime(market_date)
    power_mkt['fwd_date'] = market_date
    exposure_dates = [ds.convert_datetime_str(market_date_dt + dt.timedelta(days=d))
                      for d in exposure_day_diff]
    power_mkt['nb_sim'] = 20000

    # replacing the correlation matrix, this is a hack
    mm_corr = mrds.mrds_calib_multiple(['ATSI_2X16',
                                        'ATSI-PEAK',
                                        'ATSI_7X8',
                                        'NG_MICHCON_GD-PEAK'],
                                        '20150401', [9, 9, 9, 9])

    exposure = cva_tolling.cva_tolling(power_mkt, power_mkt['fwd_date'],
                                       params.toll_start_date,
                                       params.toll_end_date,
                                       exposure_dates,
                                       params, 20000, 20000,
                                       parallel_ind=False,
                                       cuda_ind=cuda_ind, 
                                       pricing_model_ind='skew', 
                                       new_corr_mtx=mm_corr.complete_corr_mat)
    print "Exposure =", exposure
    return True


def test_cva_rt(nb_sim=10000, ci=False, precompute=False):
    """
    real time swap CVA computation
    """
    F_0 = np.array([80., 81., 82., 81.5])
    s_0 = np.array([0.2, 0.22, 0.24, 0.22])
    rho = 0.95
    rho_m = vols.corr_hyp_sec_mat(rho, range(len(F_0)))
    tenor_t = np.array([1., 2., 6., 12.])/12.  # tenor times
    # exp_t = np.array([1, 2, 7, 14, 21, 31, 62, 120, 180, 360])/365.  # exposure times
    exp_t = np.linspace(1./365., 360./365., 50)
    swap_k = 80.
    cva_rt.disp_cva_rt(F_0, s_0, rho_m, tenor_t, exp_t, swap_k, nb_sim,
                       cuda_ind=ci,
                       cva_vals_precompute=precompute)

    return True

# test_cva_rt(ci=True)
