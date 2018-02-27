## First lines are for Python in Morgan Stanley
## see also http://wiki.ms.com/twiki/cgi-bin/view/Python/InstalledModules
import config
from numpy import *
import scipy
import sys # module for pathload
import os
import mrds # mrd skew
import pricers
import pickle
import shutil 

if __name__ == '__main__':

    # calibration section 
    reload (mrds)
    sh_name = 'WTI_JW'  # high skew
    wb_name = '/home/brumen/workspace/mrds/inputs_master.xls'
    fwd_subset_idx = arange (20) # subrange of the inputs that are calibrated 
    obj_file = '/home/brumen/workspace/mrds/mobj/obj_2.obj' 

    b = mrds.mrds_calib (wb_name, sh_name, [sh_name], ['MC_0'], 'usd_rate', \
                         [], 100, arange (20), obj_file, mt_idx=True )

    # simulation section 
    #b = pickle.load (open (obj_file) ) # loading the calibrated object
    pickle.dump (b, open (obj_file, 'wb'))

    model='skew' # CHOOSE: skew, ln_ln, ln 
    sim_times = arange (0,13)/12.0
    #sim_times = array ([0.5, 1.0]) 
    nb_sims = 100000
    b.model_skew_ln_ind = model
    b.update_sim_times (sim_times)
    b.simulate_curves(nb_sims)


    # pricing section 
    pricers.er_cs (b, [15, 1.2, 0.1, 0.03]) 
    pricers.kiko_rebate (b, [15, [0.95, 1.2],1.])

    # deltas, vegas
    print mrds.compute_partial_deltas (b, pricers.sb_gen, [ 7, 5.7, 6.5 ], 200000, array([7]), seed=123, \
                                       verbose=True) 
