## First lines are for Python in Morgan Stanley
## see also http://wiki.ms.com/twiki/cgi-bin/view/Python/InstalledModules

import config
from numpy import *
import scipy
import sys # module for pathload
import mrds # importing the MRD skew model 
import pricers
import pickle

if __name__ == '__main__':

    wb_name = 'U:/reports/cm/inputs_master.xls' # master inputs file, can be copied anywhere
    #wb_name = 'C:/Documents and Settings/brumen/cm_inputs/inputs_master.xls' 
    sh_name = 'WTI_JW'  # high skew
    sh_name_0 = 'metal_0'
    sh_name_1 = 'metal_1'
    sh_name_2 = 'metal_2'
    # sh_name = 'corn_JW' # very little skew 
    market_f = 'u:/reports/cm/reviews/basket/basket_barrier/python/mobj/b_bb_skew_2.obj' #skew_1 .... LN model, skew_2 .... skew model

    # trivariate spread cva data 
    nb_assets = 3
    # reload (mrds)
    subset_idx = arange (20) # array([12,13,14,15]) # arange (5) # subrange of the inputs that are calibrated 
    b = mrds.mrd_skew (nb_assets, multi_thread_ind = False, subset_idx=subset_idx, verbose=True)
    print "verbosity =", b.verbose
    b.read_sim_date(wb_name, sh_name) # this needs to be before others 
    b.read_curve_vol_data(0, wb_name, sh_name_0, subset_idx, arange(1))
    b.read_curve_vol_data(1, wb_name, sh_name_1, subset_idx, arange(1))
    b.read_curve_vol_data(2, wb_name, sh_name_2, subset_idx, arange(1))
    used_idx = arange(40) # use the first 30 months of discount curve 
    b.read_discount_curve(wb_name, 'usd_rate')
    b.read_model_config(0, wb_name, 'MC_0')
    b.read_model_config(1, wb_name, 'MC_1')
    b.read_model_config(2, wb_name, 'MC_2')

    b._market_corr[0][1] = b._market_corr[1][0] = ones ((1, len (subset_idx))).ravel() * 0.462
    b._market_corr[1][2] = b._market_corr[2][1] = ones ((1, len (subset_idx))).ravel() * 0.458
    b._market_corr[0][2] = b._market_corr[2][0] = ones ((1, len (subset_idx))).ravel() * 0.99


    b._set_other_params()
    # calibration part 
    b.__kappa_sigma_rho(0)
    b.__kappa_sigma_rho(1)
    b.__kappa_sigma_rho(2)
    b.calibrate_skew_params(0)
    b.calibrate_skew_params(1)
    b.calibrate_skew_params(2)
    b.generate_large_corr_mat()
    # save the calibrated market object to a file 
    pickle.dump (b, open (market_f, 'wb') )

    # loading the calibrated object
    b = pickle.load (open (market_f) )
    b.generate_large_corr_mat() # Forgot to include it before the object was saved

    model='ln_ln' # skew, ln_ln, ln 
    #sim_times = array([0, 0.5,1.0]) # arange(5)/4.0 # zero is excluded
    #sim_times = arange (6,19)/12.0
    sim_times = arange (0,13)/12.0
    print sim_times
    nb_sims = 100000

    b.model_skew_ln_ind = model
    b.update_sim_times (sim_times)
    b.simulate_curves(nb_sims)

    fwd = 15
    B = 1.2
    K = 1.0
    rebate = 0.0
    params = [ fwd, B, K, rebate]

    print pricers.bb_rebate (b, params)

    # deltas 
    print mrds.compute_partial_deltas (b, pricers.coupon_strip, params, 200000, array([12]), seed=123, \
                                       verbose=True) 

