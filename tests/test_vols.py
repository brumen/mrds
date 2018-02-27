#
# jw volatilities TESTING 

import config 
from numpy import * 
from scipy import *
#import scipy.optimize
#import scipy.integrate
#import scipy.special
import scipy.stats 
#import scipy.optimize 

import vols # class to test 
import unittest 


# jw parametrization (inherits from vol_param)
class jw_paramsTest(unittest.TestCase):
    def test_positive(self):
        value = vols.jw_params(0.2, 0.1, 0.2, 0.1, 0.02)
        print value.compute_vol (100,110,1)
        cnd = True # initial value of the condition 
        for ind in arange(90,110,1):
            cnd = cnd & (value.compute_vol (100, ind, 1) > 0)

        self.assertTrue ( cnd )

    # MISSING TESTS FOR NOT BEING INF OR NAN


class jw7_paramsTest (unittest.TestCase):
    # def test_positive(self):
    #     value = vols.jw7_params(0.2, 0.1, 0.2, 0.1, 0.02, 0.1, 0.2)
    #     cnd_vols = True # initial value of the condition for vols checking 
    #     cnd_lv = True # init for local vol
    #     for ind in arange(90,110,1):
    #         cnd_vols = cnd_vols & (value.compute_vol (100, ind, 1) > 0)
    #         cnd_lv = cnd_lv & value.local_vol_jw7_alt ( ind, S, S_0, ttm, disc_fact, T ):
    #     self.assertTrue ( cnd )


    # checks whether the implied vols are positive (simple test)
    def test_implied_vol (self):
        #value = vols.jw7_params (array([[100., 0.2, 1.5, 0.5, 0.1, 0.12, 0.1, 0.2]]))
        value = vols.jw7_params (array([[100., 0.2, 1.5, 0.5, 0.1, 0.12, 0.1, 0.2], 
                                        [100., 0.2, 0.5, 1.5, 0.1, 0.12, 0.1, 0.2]]))
        cnd_iv = True # conditional of the implied vol
        K_range = arange (80., 130., 1.)
        imp_v_range = zeros (len (K_range) )
        for K_ind, K in enumerate(K_range):
            curr_vol = value.implied_vol (1, K, 0.3)
            imp_v_range[K_ind] = curr_vol
            cnd_iv = cnd_iv & (curr_vol > 0)

        print "Implied vols = ", imp_v_range

        self.assertEqual (cnd_iv,True)


    def test_implied_vol_all_fwd (self):
        #value = vols.jw7_params (array([[100., 0.2, 1.5, 0.5, 0.1, 0.12, 0.1, 0.2]]))
        value = vols.jw7_params (array([[100., 0.2, 1.5, 0.5, 0.1, 0.12, 0.1, 0.2], 
                                        [100., 0.2, 0.5, 1.5, 0.1, 0.12, 0.1, 0.2]]))
        cnd_iv = True # conditional of the implied vol
        K_range = arange (80., 130., 1.)
        
        curr_vol = value.implied_vol_all_fwd (K_range, array([0.4]) )
        curr_vol_pos = curr_vol > 0.
        cnd_iv = curr_vol_pos.all()

        print "Implied vols = ", curr_vol

        self.assertEqual (cnd_iv,True)

        
    # tests the functioning of the skew distribution 
    def test_skew_distr (self):
        #value = vols.jw7_params (array([[100., 0.2, 1.5, 0.5, 0.1, 0.12, 0.1, 0.2]]))
        value = vols.jw7_params (array([[100., 0.2, 1.5, 0.5, 0.1, 0.12, 0.1, 0.2], 
                                        [100., 0.2, 0.5, 1.5, 0.1, 0.12, 0.1, 0.2]]))
        cnd_iv = True # conditional of the implied vol

        print value.call_future_T (0, 100., 110., 1.)
        print value.call_future_K (0, 100., 110., 1.)
        print value.call_future_KK (0, 100., 110., 1.)
        print value.skewed_distribution (0, 100., 101., 1.)
        print value.skewed_cdf (0, 100., 100., 1., 1.)
        print value.inversion_skewed_cdf (0, 100., 1., 2.5)

        self.assertEqual (cnd_iv,True)


    def test_skew_distr_speed (self):
        #value = vols.jw7_params (array([[100., 0.2, 1.5, 0.5, 0.1, 0.12, 0.1, 0.2]]))
        value = vols.jw7_params (array([[100., 0.2, 1.5, 0.5, 0.1, 0.12, 0.1, 0.2], 
                                        [100., 0.2, 0.5, 1.5, 0.1, 0.12, 0.1, 0.2]]))
        cnd_iv = True # conditional of the implied vol

        quantile_range = arange(0.1, 1., 0.1)
        for quantile in quantile_range: 
            qv = value.inversion_skewed_cdf (0, 100., 1., quantile)
            print "quantile = ", quantile, " solu = ", qv


        self.assertEqual (cnd_iv,True)



    

# running the above tests 
suite = unittest.TestLoader().loadTestsFromTestCase(jw7_paramsTest)
unittest.TextTestRunner(verbosity=2).run(suite)
