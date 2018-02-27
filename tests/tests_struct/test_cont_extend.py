 ## First lines are for Python in Morgan Stanley
## see also http://wiki.ms.com/twiki/cgi-bin/view/Python/InstalledModules

import sys
#sys.path.append('U:/projects/eclipse_work/mrds/src/')
import config
from numpy import *
import scipy
import sys # module for pathload
import os
# graphics packages needed
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
# importing the MRD skew model 
import mrds
import pricers
import pickle
import shutil 

if __name__ == '__main__':

    wb_name = config.market_file
    sh_name = 'WTI_JW'  # high skew
    obj_file = '/home/brumen/workspace/mrds/tmp/wti_skew.obj' # 
    nb_assets = 1
    subset_idx = arange (25) # array([12,13,14,15]) # arange (5) # subrange of the inputs that are calibrated 
    nb_sims = 100000

    mrds.mrds_calib (wb_name, sh_name, [sh_name], ['MC_0'], 'usd_rate', \
                     [], nb_sims, subset_idx, obj_file, mt_idx=True) #, verb_idx='high')


    b = pickle.load (open (obj_file) ) # loading the calibrated object
        
    model='skew' # CHOOSE: skew, ln_ln, ln 
    sim_times = arange (0,5)/10. #array([0., 0.1, 0.2, 0.3, 0.4, 0.5]) # half a year extension 
    #sim_times = array ([0., 1.]) 
    print sim_times
    nb_sims = 10000
    b.model_skew_ln_ind = model
    b.update_sim_times (sim_times)
    b.simulate_curves(nb_sims)

    # contingent extendible testing 
    ext_mat, swap_mat = [0.5, 1.0] 
    [ swap_K, apo_c_K, apo_p_K ] = [ 90., 120., 58. ]      # strikes 
    swap_nb, apo_c_nb, apo_p_nb = [ 1., -0., 1.]    # amounts 
    beta, sigma_L, corr = [1.05, 1.8, 0.99]     # samuelson params
    #beta, sigma_L, corr = [0.01, 0.01, 0.99]
    params = [[[ext_mat, swap_mat], [swap_mat, 1.5]],[[swap_K, 100.], [apo_c_K]*2, [apo_p_K]*2],
              [[swap_nb, swap_nb], [apo_c_nb]*2, [apo_p_nb]*2],
              [beta, sigma_L], -1, corr ]

    params2 = [[[ext_mat], [swap_mat]],[[swap_K], [apo_c_K], [apo_p_K]],
               [[swap_nb], [apo_c_nb], [apo_p_nb]],
               [beta, sigma_L], -1, corr ]


    print pricers.cont_extend (b, params2) # prices 


    print pricers.pricer_multiple ( b, pricers.cont_extend, params, nb_sims, 10 )



    # deltas
    print mrds.compute_partial_deltas (b, pricers.cont_extend, params, \
                                       20000, array([7]), seed=123, \
                                       verbose=True) 
    
