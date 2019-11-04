# Unit Testing script of mrds module 
# 

from numpy import exp, sqrt, pi, inf, arange, array
from numpy.linalg.linalg import norm
import scipy
import scipy.optimize
from scipy.integrate import quad

import pricers
import unittest
import pickle
import vols


class ComSkewTests(unittest.TestCase):

    # reads the market object 
    def setUp (self):
        # TODO: FIX THIS CONSIDERABLY
        self.mo = pickle.load (open ('/home/brumen/workspace/mrds/mobj/wti_skew.obj') ) # loading the calibrated object

    def test_trunc_normal_above(self):
        """ Tests the _trunc_normal_above function of the ComSkewMaths class.
        """

        for a in arange(-3, 3, 0.1):
            mo_res = self.mo._trunc_normal_above(a)
            num_res = array([ quad (lambda x: 1. / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0],
                              quad (lambda x: x / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0],
                              quad (lambda x: x**2 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0],
                              quad (lambda x: x**3 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0],
                              quad (lambda x: x**4 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0] ] )

            self.assertTrue(norm ( mo_res - num_res) < 1e-7 )

    def test_trunc_normal_below(self):

        for a in arange (-3, 3, 0.1):
            mo_res = self.mo._trunc_normal_below(a)
            num_res = array([ quad (lambda x: 1. / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0],
                              quad (lambda x: x / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0],
                              quad (lambda x: x**2 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0],
                              quad (lambda x: x**3 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0],
                              quad (lambda x: x**4 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0] ] )
            self.assertTrue( norm ( mo_res - num_res) < 1e-7 )

    def test_trunc_normal_interval(self):

        for a in arange (-3,3,0.1):
            mo_res = self.mo._trunc_normal_interval(a, a + 1.)
            num_res = array([ quad (lambda x: 1. / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0],
                              quad (lambda x: x / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0],
                              quad (lambda x: x**2 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0],
                              quad (lambda x: x**3 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0],
                              quad (lambda x: x**4 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0] ] )
            self.assertTrue( norm ( mo_res - num_res) < 1e-7 )

    def test_inverse_naive (self):
        """ Comparing the black_vol_inverse: advanced function and _naive function.
        """

        F = 100
        K = 100.0
        p = 21
        dt = 1
        DF = 0.99
        theta = 1
        tol = 1e-8
        inv1 = array([vols.black_vol_inverse_naive(F, K, p, dt, DF, theta, tol) for K in range (80,120,1)])
        inv2 = array([vols.black_vol_inverse(F, K, p, dt, DF, theta, tol) for K in range (80,120,1)])
        
        res = reduce (lambda x,y: x and y, abs (inv1 - inv2) < 1e-5, True)
        self.assertTrue ( res )

    def test_negative_forwards(self):
        """ Tests whether the model simulation produces negative prices
        """
        model='skew' # CHOOSE: skew, ln_ln, ln
        sim_times = arange (0,5)/10. #array([0., 0.1, 0.2, 0.3, 0.4, 0.5]) # half a year extension 
        # sim_times = array ([0., 1.]) 
        nb_sims = 100000
        self.mo.model_skew_ln_ind = model
        self.mo.update_sim_times (sim_times)
        self.mo.simulate_curves(nb_sims)
        skew_ind = sum (self.mo.simulated_curves[0] < 0 ) < 50 

        # TODO: THIS SHOULD BE THE NEXT CASE
        model='ln_ln' # CHOOSE: skew, ln_ln, ln
        sim_times = arange (0,5)/10. #array([0., 0.1, 0.2, 0.3, 0.4, 0.5]) # half a year extension 
        nb_sims = 100000
        self.mo.model_skew_ln_ind = model
        self.mo.update_sim_times (sim_times)
        self.mo.simulate_curves(nb_sims)
        ln_ln_ind = sum (self.mo.simulated_curves[0] < 0 ) < 50 

        #
        # print "Log-normal (LN simulation): Testing the non-negativity of futures prices"
        model='ln' # CHOOSE: skew, ln_ln, ln 
        sim_times = arange (0,5)/10. #array([0., 0.1, 0.2, 0.3, 0.4, 0.5]) # half a year extension 
        nb_sims = 100000
        self.mo.model_skew_ln_ind = model
        self.mo.update_sim_times (sim_times)
        self.mo.simulate_curves(nb_sims)
        ln_ind = sum (self.mo.simulated_curves[0] < 0 ) < 50 

        self.assertTrue ( skew_ind and ln_ln_ind and ln_ind )

    def test_martingale (self):
        """ Tests whether the futures prices are martingales (at 0)
        """

        model='skew' # CHOOSE: skew, ln_ln, ln
        sim_times = arange (0,12)/12. # 1 year, monthly simulations
        nb_sims = 1000000
        self.mo.model_skew_ln_ind = model
        self.mo.update_sim_times (sim_times)
        self.mo.simulate_curves(nb_sims)

        nb_cms = len (self.mo.simulated_curves) # nb_sims. of commodities
        skew_ind = [False] * nb_cms # placeholder
        for cm in range (nb_cms): # do this for every commodity
            for fwd in range ( sum (self.mo.option_tenors_list[cm] <= sim_times[-1]), 
                               len (self.mo.forward_curve_list[cm]) ):
                # print "Skew: Testing the martingale of futures prices for com.", cm," and future nb_sims. ", fwd
                skew_ind[cm] = scipy.linalg.norm ( mean ( self.mo.simulated_curves[cm][:,fwd,:], axis=1) - 
                                                   self.mo.simulated_curves[cm][0,fwd,0] ) < 1e-3
                
                # print "Simulated values = ", mean ( self.mo.simulated_curves[cm][:,fwd,:], axis=1)
                # print "Theoretical values = ", self.mo.simulated_curves[cm][0,fwd,0]

        self.assertTrue ( reduce (lambda x,y: x and y, skew_ind, True) ) # all skew indicators have to be 0

    def test_poly_eu_no_roots (self):
        """ Testing the roots part of _polynomial_european"
        """
        C1_range = arange (-1., 1., 0.05)
        C2_range = arange (-2., 2., 0.05)
        C3_range = arange (-2., 4., 0.05)
        var_C = self.mo.C_vec_list[0][11]
        K = 130.
        assert1 = True
        assert2 = True
        assert3 = True

        # CALL option testing 
        # C1 testing of the poly. european 
        for C1 in C1_range:
            var_C_cur = array([C1, var_C[1], var_C[2]])
            self.mo.debug_mode = False
            res1 = self.mo._polynomial_european(0, var_C_cur, 11, K, 1)
            self.mo.debug_mode = True
            res2 = self.mo._polynomial_european(0, var_C_cur, 11, K, 1)
            assert1 = assert1 and ( norm ( res1 - res2) < 1e-6 )


        for C2 in C2_range:
            var_C_cur = array([var_C[0], C2, var_C[2]])
            self.mo.debug_mode = False
            res1 = self.mo._polynomial_european(0, var_C_cur, 11, K, 1)
            self.mo.debug_mode = True
            res2 = self.mo._polynomial_european(0, var_C_cur, 11, K, 1)
            assert2 = assert2 and ( norm ( res1 - res2) < 1e-6 )


        for C3 in C3_range:
            var_C_cur = array([var_C[0], var_C[1], C3])
            self.mo.debug_mode = False
            res1 = self.mo._polynomial_european(0, var_C_cur, 11, K, 1)
            self.mo.debug_mode = True
            res2 = self.mo._polynomial_european(0, var_C_cur, 11, K, 1)
            assert3 = assert3 and ( norm ( res1 - res2) < 1e-6 )


        # PUT option testing
        K = 70.
        assert4 = True
        assert5 = True
        assert6 = True

        # C1 testing of the poly. european 
        for C1 in C1_range:
            var_C_cur = array([C1, var_C[1], var_C[2]])
            self.mo.debug_mode = False
            res1 = self.mo._polynomial_european(0, var_C_cur, 11, K, -1)
            self.mo.debug_mode = True
            res2 = self.mo._polynomial_european(0, var_C_cur, 11, K, -1)
            assert4 = assert4 and ( norm ( res1 - res2) < 1e-6 )


        for C2 in C2_range:
            var_C_cur = array([var_C[0], C2, var_C[2]])
            self.mo.debug_mode = False
            res1 = self.mo._polynomial_european(0, var_C_cur, 11, K, -1)
            self.mo.debug_mode = True
            res2 = self.mo._polynomial_european(0, var_C_cur, 11, K, -1)
            assert5 = assert5 and ( norm ( res1 - res2) < 1e-6 )


        for C3 in C3_range:
            var_C_cur = array([var_C[0], var_C[1], C3])
            self.mo.debug_mode = False
            res1 = self.mo._polynomial_european(0, var_C_cur, 11, K, -1)
            self.mo.debug_mode = True
            res2 = self.mo._polynomial_european(0, var_C_cur, 11, K, -1)
            assert6 = assert6 and ( norm ( res1 - res2) < 1e-6 )

        self.assertTrue(assert1 and assert2 and assert3 and assert4 and assert5 and assert6)

    def test_ln_model(self):
        sim_times = array([0.5, 1.])
        nb_sims = 10000000
        self.mo.model_skew_ln_ind = 'ln_ln'
        self.mo.update_sim_times (sim_times)
        self.mo.simulate_curves(nb_sims)

        pr1 = pricers.call (self.mo, [self.mo.forward_curve_list[0][15], 
                                      15, 1])

        disc_fact = self.mo.DF (1.)
        theta = 1 # call options 
        iv = vols.black_vol_inverse_vec (self.mo.forward_curve_list[0][15], 
                                         [self.mo.forward_curve_list[0][15]], 
                                         [pr1], 1., disc_fact, theta, 1e-4)

        self.assertTrue(abs (iv - self.mo.atm_vol_list[0][15] ) < 1e-4)
