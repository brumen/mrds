#
# tests the functions regarding the tolling model
#

import datetime
import numpy as np

from unittest import TestCase

from mrds.tolling.tolling import TollingModel


class TestTolling(TestCase):

    def test_smoke(self):
        """ Run smoke test on the tolling model.
        """

        tm = TollingModel.from_db( datetime.date(2015, 4, 1)
                                 , datetime.date(2015, 5, 1)
                                 , datetime.date(2015, 6, 20)
                                 , ['WTI', 'BRENT', ]
                                 , ['WTI', 'BRENT', ]
                                 , {'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,)}
                                 , { 'WEEKDAY': [('WTI', 8), ('BRENT', 16), ]
                                   , 'WEEKEND': [('WTI', 16), ('BRENT', 8), ]}
                                 , 'BRENT'
                                 , ['WTI', 'BRENT', ] )

        tm.dispatch_all(tm.toll_start, tm.toll_end )

    def test_tensor_mt(self):
        s1 = 10
        s2 = 2000
        P = np.random.random(size=(s1, s1, s2, s2))
        H = np.random.random(size=(s2, s2))
        G = np.random.random(size=(s2, s2))
        res1 = np.zeros((s1, s1))

        tolling.tens_fast_mt(P, H, G, res1)

        res2 = np.zeros((s1, s1))
        for i1 in range(s1):
            for i2 in range(s1):
                res2[i1, i2] = np.sum(P[i1, i2, :, :] * H) + G[i1, i2]

        return True

    def test_1(self):
        F_power = np.linspace(30., 30+5., 12)
        F_gas = np.linspace(4., 6., 12)
        T_l = np.arange(12)/12.  # monthly sims
        vol_power = np.linspace(0.2, 0.3, 12)
        vol_gas = vol_power
        cash_vol = np.linspace(1.2,1.3,12)
        HR = 8.
        mm_sim_nb = 1000
        pm_sim_nb = 18000

        revs = pricers.tolling_simple_nocuda(F_power, F_gas,
                                             T_l, vol_power, vol_gas,
                                             cash_vol, HR, mm_sim_nb,
                                             pm_sim_nb)
        print ("revs = ", np.mean(revs, axis=0))

        self.assertTrue(True)

    def test_2(self):
        F_power = np.linspace(30., 30+5., 12)
        F_gas = np.linspace(4., 6., 12)
        T_l = np.arange(12)/12.  # monthly sims
        vol_power = np.linspace(0.2, 0.3, 12)
        vol_gas = vol_power
        cash_vol = np.linspace(1.2, 1.3, 12)
        HR = 8.
        mm_sim_nb = 1000
        pm_sim_nb = 65000

        revs = pricers.tolling_simple_cuda(F_power, F_gas,
                                           T_l, vol_power, vol_gas,
                                           cash_vol, HR, mm_sim_nb,
                                           pm_sim_nb)

        print ('revs = ', np.mean(revs, axis=0))

        self.assertTrue(True)

    def test_set_params_fiction():
        params = tolling.tolling_params()
        params.cuda = False
        params.tolling_fast = False
        params.tolling_fast_mt = False
        params.lattice_size = 60
        params.MDT = 3  # in number of blocks (appx 4 - 24 hours)
        params.MUT = 3
        params.maxStarts = 1000  # NOT YET IMPLEMENTED
        params.maxCap = 22.47
        params.minCap = 10.34
        params.fixedStartupCost = 4500.
        params.fixedStartupCostCold = params.fixedStartupCost  # NOT YET IMPLEMENTED
        params.fixedShutdownCost = 0.
        params.rampDnCost = 0.
        params.rampUpCost = 0.

        market_v = tolling.tolling_params()
        market_v.nb_months = 10
        market_v.Kv = np.linspace(43., 47., 10)
        market_v.Fv = np.linspace(45., 50., 10)
        market_v.Fv_OP = np.linspace(45., 50., 10)  # off-peak
        market_v.Fv_P = np.linspace(50., 60., 10)  # peak
        market_v.T_mv = np.linspace(2./12., 1, 10)

        market_v.sigma_Fv = np.linspace(0.2, 0.4, 10)
        market_v.sigma_Fv_OP = np.linspace(0.2, 0.4, 10)
        market_v.sigma_Fv_P = np.linspace(0.3, 0.5, 10)

        market_v.sigma_Cv_OP = np.array([1.3])
        market_v.sigma_Cv_P = market_v.sigma_Cv_OP

        blocks = tolling.tolling_params()
        blocks.days_partition = [[0, 1, 2, 3, 4], [5, 6]]
        blocks.hours_partition = [[6, 18], [12, 12]]
        blocks.market = [[{"name": "peak",
                           "fwd": market_v.Fv_P[0],
                           "sigma_F": market_v.sigma_Fv_P[0],
                           "sigma_C": market_v.sigma_Cv_P[0]},
                          {"name": "offpeak",
                           "fwd": market_v.Fv_OP[0],
                           "sigma_F": market_v.sigma_Fv_OP[0],
                           "sigma_C": market_v.sigma_Cv_OP[0]}],
                         [{"name": "w_peak",
                           "fwd": market_v.Fv_wOP[0],
                           "sigma_F": market_v.sigma_Fv_OP[0],
                           "sigma_C": market_v.sigma_Cv_OP[0]},
                          {"name": "w_offpeak",
                           "fwd": market_v.Fv_OP[0],
                           "sigma_F": market_v.sigma_Fv_OP[0],
                           "sigma_C": market_v.sigma_Cv_OP[0]}]]  # corresponds to hours partition

        blocks_m = []
        for m in range(10):
            blocks_tmp = tolling.tolling_params()
            blocks_tmp.days_partition = [[0, 1, 2, 3, 4], [5, 6]]
            blocks_tmp.hours_partition = [[6, 18], [6, 18]]
            blocks_tmp.K = market_v.Kv[m]
            blocks_tmp.Tm = market_v.T_mv[m]
            blocks_tmp.market = [[{"name": "peak",
                                   "fwd": market_v.Fv_P[m],
                                   "sigma_F": market_v.sigma_Fv_P[m],
                                   "sigma_C": market_v.sigma_Cv_P[m]},
                                  {"name": "offpeak",
                                   "fwd": market_v.Fv_OP[m],
                                   "sigma_F": market_v.sigma_Fv_OP[m],
                                   "sigma_C": market_v.sigma_Cv_OP[m]}],
                                 [{"name": "offpeak",
                                   "fwd": market_v.Fv_OP[m],
                                   "sigma_F": market_v.sigma_Fv_OP[m],
                                   "sigma_C": market_v.sigma_Cv_OP[m]},
                                  {"name": "offpeak",
                                   "fwd": market_v.Fv_OP[m],
                                   "sigma_F": market_v.sigma_Fv_OP[m],
                                   "sigma_C": market_v.sigma_Cv_OP[m]}]]
            blocks_m.append(blocks_tmp)

        return params, market_v, blocks, blocks_m


    def test_params_monthly_toll(nb_months=1):
        params = tolling.tolling_params()
        params.cuda = False
        params.tolling_fast = False
        params.tolling_fast_mt = False
        params.lattice_size = 2
        params.MDT = 3  # in number of blocks (appx 4 - 24 hours)
        params.MUT = 3
        params.maxStarts = 1000  # NOT YET IMPLEMENTED
        params.maxCap = 22.47
        params.minCap = 10.34
        params.fixedStartupCost = 4500.
        params.fixedStartupCostCold = params.fixedStartupCost  # NOT YET IMPLEMENTED
        params.fixedShutdownCost = 0.
        params.rampDnCost = 0.
        params.rampUpCost = 0.

        market_v = tolling.tolling_params()
        market_date = dt.datetime(2014, 3, 11)  # 3/11/2014
        forward_date = dt.datetime(2016, 5, 1)  # 1/1/2016
        market_v.T_mv = np.array([(forward_date-market_date).days/365.])  # np.array([1.81])
        market_v.nb_months = 1
        market_v.Kv = np.array([42.34])
        market_v.Fv_P = np.array([40.79])  # ([43.47]) # peak, weekday
        market_v.Fv_wOP = np.array([36.88])  # ([38.39]) # off-peak, weekend
        market_v.Fv_OP = np.array([21.49])  # ([31.74]) # off-peak, weekday

        market_v.sigma_Fv_OP = np.array([0.1503])  # ([0.1859])
        market_v.sigma_Fv_P = market_v.sigma_Fv_OP

        market_v.sigma_Cv_OP = np.array([1.05])  # ([1.3])
        market_v.sigma_Cv_P = market_v.sigma_Cv_OP

        blocks = tolling.tolling_params()
        blocks.days_partition = [[0, 1, 2, 3, 4], [5, 6]]
        blocks.hours_partition = [[6, 18], [6, 18]]
        blocks.market = [[{"name": "peak",
                           "fwd": market_v.Fv_P[0],
                           "sigma_F": market_v.sigma_Fv_P[0],
                           "sigma_C": market_v.sigma_Cv_P[0]},
                          {"name": "offpeak",
                           "fwd": market_v.Fv_OP[0],
                           "sigma_F": market_v.sigma_Fv_OP[0],
                           "sigma_C": market_v.sigma_Cv_OP[0]}],
                         [{"name": "w_peak",
                           "fwd": market_v.Fv_wOP[0],
                           "sigma_F": market_v.sigma_Fv_OP[0],
                           "sigma_C": market_v.sigma_Cv_OP[0]},
                          {"name": "w_offpeak",
                           "fwd": market_v.Fv_OP[0],
                           "sigma_F": market_v.sigma_Fv_OP[0],
                           "sigma_C": market_v.sigma_Cv_OP[0]}]]  # corresponds to hours partition

        blocks_m = []
        for m1 in range(nb_months):
            m = 0
            blocks_tmp = tolling.tolling_params()
            blocks_tmp.days_partition = [[0, 1, 2, 3, 4], [5, 6]]
            blocks_tmp.hours_partition = [[6, 18], [6, 18]]
            blocks_tmp.K = market_v.Kv[m]
            blocks_tmp.Tm = market_v.T_mv[m]
            blocks_tmp.market = [[{"name": "peak",
                                   "fwd": market_v.Fv_P[m],
                                   "sigma_F": market_v.sigma_Fv_P[m],
                                   "sigma_C": market_v.sigma_Cv_P[m]},
                                  {"name": "offpeak",
                                   "fwd": market_v.Fv_OP[m],
                                   "sigma_F": market_v.sigma_Fv_OP[m],
                                   "sigma_C": market_v.sigma_Cv_OP[m]}],
                                 [{"name": "offpeak",
                                   "fwd": market_v.Fv_OP[m],
                                   "sigma_F": market_v.sigma_Fv_OP[m],
                                   "sigma_C": market_v.sigma_Cv_OP[m]},
                                  {"name": "offpeak",
                                   "fwd": market_v.Fv_OP[m],
                                   "sigma_F": market_v.sigma_Fv_OP[m],
                                   "sigma_C": market_v.sigma_Cv_OP[m]}]]
            blocks_m.append(blocks_tmp)

        return params, market_v, blocks, blocks_m


    def test_params_gas_toll():
        params = tolling.tolling_params()
        params.cuda = False
        params.fuel_ind = True
        params.tolling_fast = False
        params.tolling_fast_mt = False
        params.lattice_size = 50
        params.MDT = 3  # in number of blocks (appx 4 - 24 hours)
        params.MUT = 3
        params.maxStarts = 1000  # NOT YET IMPLEMENTED
        params.maxCap = 22.47
        params.minCap = 10.34
        params.fixedStartupCost = 4500.
        params.fixedStartupCostCold = params.fixedStartupCost  # NOT YET IMPLEMENTED
        params.fixedShutdownCost = 0.
        params.rampDnCost = 0.
        params.rampUpCost = 0.
        params.HR = 7.

        market_v = tolling.tolling_params()
        market_date = dt.datetime(2014, 3, 11)  # 3/11/2014
        forward_date = dt.datetime(2016, 5, 1)  # 1/1/2016
        market_v.T_mv = np.array([(forward_date-market_date).days/365.])  # np.array([1.81])
        market_v.nb_months = 1
        market_v.Kv = np.array([42.34])
        market_v.Fv_P = np.array([40.79])  # ([43.47]) # peak, weekday
        market_v.Fv_wOP = np.array([36.88])  # ([38.39]) # off-peak, weekend
        market_v.Fv_OP = np.array([21.49])  # ([31.74]) # off-peak, weekday

        market_v.sigma_Fv_OP = np.array([0.1503])  # ([0.1859])
        market_v.sigma_Fv_P = market_v.sigma_Fv_OP

        market_v.sigma_Cv_OP = np.array([1.05])  # ([1.3])
        market_v.sigma_Cv_P = market_v.sigma_Cv_OP

        market_v.Gv = np.array([4.])
        market_v.sigma_Gv = np.array([0.3])
        market_v.sigma_Gv_C = np.array([0.9])

        blocks = { 'days_partition'  : [[0, 1, 2, 3, 4], [5, 6]]
                 , 'hours_partition' : [[6, 18], [6, 18]] }

        market = {'market': {'power': [[{"name": "peak",
                                 "fwd": market_v.Fv_P[0],
                                 "sigma_F": market_v.sigma_Fv_P[0],
                                 "sigma_C": market_v.sigma_Cv_P[0]},
                                {"name": "offpeak",
                                 "fwd": market_v.Fv_OP[0],
                                 "sigma_F": market_v.sigma_Fv_OP[0],
                                 "sigma_C": market_v.sigma_Cv_OP[0]}],
                               [{"name": "w_peak",
                                 "fwd": market_v.Fv_wOP[0],
                                 "sigma_F": market_v.sigma_Fv_OP[0],
                                 "sigma_C": market_v.sigma_Cv_OP[0]},
                                {"name": "w_offpeak",
                                 "fwd": market_v.Fv_OP[0],
                                 "sigma_F": market_v.sigma_Fv_OP[0],
                                 "sigma_C": market_v.sigma_Cv_OP[0]}]]  # corresponds to hours partition
                          , 'gas' : {"fwd": market_v.Gv,
                                     "sigma_F": market_v.sigma_Gv,
                                     "sigma_C": market_v.sigma_Gv_C} # monthly, cash vol for gas
                          }}
        blocks_m = []
        blocks_tmp = tolling.tolling_params()
        for m in range(1):
            blocks_tmp.days_partition = [[0, 1, 2, 3, 4], [5, 6]]
            blocks_tmp.hours_partition = [[6, 18], [6, 18]]
            blocks_tmp.K = market_v.Kv[m]
            blocks_tmp.Tm = market_v.T_mv[m]
            blocks_tmp.market = tolling.tolling_params()
            blocks_tmp.market.power = [[{"name": "peak",
                                         "fwd": market_v.Fv_P[m],
                                         "sigma_F": market_v.sigma_Fv_P[m],
                                         "sigma_C": market_v.sigma_Cv_P[m]},
                                        {"name": "offpeak",
                                         "fwd": market_v.Fv_OP[m],
                                         "sigma_F": market_v.sigma_Fv_OP[m],
                                         "sigma_C": market_v.sigma_Cv_OP[m]}],
                                       [{"name": "offpeak",
                                         "fwd": market_v.Fv_OP[m],
                                         "sigma_F": market_v.sigma_Fv_OP[m],
                                         "sigma_C": market_v.sigma_Cv_OP[m]},
                                        {"name": "offpeak",
                                         "fwd": market_v.Fv_OP[m],
                                         "sigma_F": market_v.sigma_Fv_OP[m],
                                         "sigma_C": market_v.sigma_Cv_OP[m]}]]
            blocks_tmp.market.gas = {"fwd": market_v.Gv[m],
                                     "sigma_F": market_v.sigma_Gv[m],
                                     "sigma_C": market_v.sigma_Gv_C[m]}
            blocks_m.append(blocks_tmp)

        return params, market_v, blocks, blocks_m


    def test_monthly_tolling(nb_months=1, mp_ind=False, cuda_ind=False):
        """
        tests the monthly tolling model
        """
        # nb_months = 1
        params, market_v, blocks, blocks_m = test_params_monthly_toll(nb_months=nb_months)
        tl2 = tolling.tolling_model_lattice_all(params, blocks_m, nb_months,
                                                keep_dec=True,
                                                cuda_ind=cuda_ind)
        tv = tl2.compute_val(mp_ind=mp_ind)

        return True


    # def test_gas_tolling():
    #    params, market_v, blocks, blocks_m = test_params_gas_toll()
    #    tl2 = tolling_gas.tolling_model_lattice_gas_all(params, blocks_m, 1)
    #    tv = tl2.compute_val()
    #    print "TV = ", tv
    #    return True



# running the above tests 
# suite = unittest.TestLoader().loadTestsFromTestCase(TollingSimpleTest)
# unittest.TextTestRunner(verbosity=2).run(suite)
# test_monthly_tolling()
