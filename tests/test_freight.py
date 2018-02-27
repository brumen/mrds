# test for the freight model
import config
import numpy as np
import freight

fwd_date = '20150401'
T = 4  # nb. time periods
locs = ['AMS', 'NYC', 'MIA', 'LA', 'SHA']
N_init = np.array([3, 4, 1, 1, 8])  # initial distribution of tankers
fwd_curves = {'AMS': np.array([95., 96., 97., 98.]),
              'NYC': np.array([92., 93., 94., 95.]),
              'MIA': np.array([91., 92., 93., 94.]),
              'LA': np.array([90., 91., 95., 100.]),
              'SHA': np.array([85., 90., 95., 100.])}

vol_curves = {'AMS': np.array([0.3, 0.32, 0.35, 0.4]),
              'NYC': np.array([0.3, 0.32, 0.35, 0.4]),
              'MIA': np.array([0.3, 0.32, 0.35, 0.4]),
              'LA': np.array([0.3, 0.32, 0.35, 0.4]),
              'SHA': np.array([0.3, 0.32, 0.35, 0.4])}

# correlation matrix
corr_mtx = {('AMS', 'AMS'): 0.98,
            ('AMS', 'NYC'): 0.9,
            ('AMS', 'MIA'): 0.95,
            ('AMS', 'LA'): 0.99,
            ('AMS', 'SHA'): 0.8,
            ('NYC', 'NYC'): 0.98,
            ('NYC', 'MIA'): 0.97,
            ('NYC', 'LA'): 0.82,
            ('NYC', 'SHA'): 0.74,
            ('MIA', 'MIA'): 0.98,
            ('MIA', 'LA'): 0.89,
            ('MIA', 'SHA'): 0.91,
            ('LA', 'LA'): 0.99,
            ('LA', 'SHA'): 0.90,
            ('SHA', 'SHA'): 0.98}

# amount of time to get from one location to the other
travel_mtx = {('AMS', 'NYC'): 1,
              ('AMS', 'MIA'): 1,
              ('AMS', 'LA'): 2,
              ('AMS', 'SHA'): 5,
              ('NYC', 'MIA'): 1,
              ('NYC', 'LA'): 3,
              ('NYC', 'SHA'): 6,
              ('MIA', 'LA'): 2,
              ('MIA', 'SHA'): 5,
              ('LA', 'SHA'): 3}

b = freight.Freight(fwd_date, locs, fwd_curves, vol_curves,
                    corr_mtx, travel_mtx, N_init, T)
#  b.set_n_solve()
