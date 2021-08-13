""" Basic tests of the ComSkew class
"""

import datetime
import scipy
import scipy.optimize
import numpy as np

from numpy               import exp, sqrt, pi, inf, arange, array
from numpy.linalg.linalg import norm
from scipy.integrate     import quad
from unittest            import TestCase

from mrds.mrds            import ComSkew
from mrds.vols.vols_basic import black_vol_inverse_vec
# from mrds.pricers.pricers import call


class ComSkewTests(TestCase):

    MKT_DATE = datetime.date(2015, 4, 1)

    m_1 = ComSkew.from_db(MKT_DATE, ['WTI'])

    def test_trunc_normal_above(self):
        """ Tests the _trunc_normal_above function of the ComSkewMaths class.
        """

        for a in arange(-3, 3, 0.1):
            mo_res = self.m_1._trunc_normal_above(a)
            num_res = array([ quad (lambda x: 1. / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0],
                              quad (lambda x: x / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0],
                              quad (lambda x: x**2 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0],
                              quad (lambda x: x**3 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0],
                              quad (lambda x: x**4 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), -inf, a)[0] ] )

            self.assertLess(norm ( mo_res - num_res), 1e-7 )

    def test_trunc_normal_below(self):

        for a in arange (-3, 3, 0.1):
            mo_res = self.m_1._trunc_normal_below(a)
            num_res = array([ quad (lambda x: 1. / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0],
                              quad (lambda x: x / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0],
                              quad (lambda x: x**2 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0],
                              quad (lambda x: x**3 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0],
                              quad (lambda x: x**4 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, inf)[0] ] )
            self.assertLess( norm ( mo_res - num_res), 1e-7 )

    def test_trunc_normal_interval(self):

        for a in arange (-3,3,0.1):
            mo_res = self.m_1._trunc_normal_interval(a, a + 1.)
            num_res = array([ quad (lambda x: 1. / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0],
                              quad (lambda x: x / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0],
                              quad (lambda x: x**2 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0],
                              quad (lambda x: x**3 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0],
                              quad (lambda x: x**4 / sqrt ( 2. * pi ) * exp ( - x**2 / 2.0 ), a, a+1.)[0] ] )
            self.assertTrue( norm ( mo_res - num_res) < 1e-7 )

    def test_negative_forwards(self):
        """ Tests whether the model simulation produces negative prices
        """

        sim_times = arange(0, 5)/10.
        nb_sims = 100000

        sim_curves = self.m_1.simulate_curves(['WTI'], nb_sims, sim_times, tenor_list = LLL)
        self.assertLess(sum (sim_curves[0]) < 0, 50)

    def test_martingale (self):
        """ Tests whether the futures prices are martingales (at 0)
        """

        sim_times = arange(0, 12)/12.
        nb_sims = 100000

        sim_curves = self.m_1.simulate_curves(['WTI'], nb_sims, sim_times, tenor_list = LLL)

        nb_cms = len (sim_curves)
        for cm in range (nb_cms): # do this for every commodity
            for fwd in range ( sum (self.m_1.option_tenors_list[cm] <= sim_times[-1]),
                               len (self.m_1.forward_curve_list[cm]) ):
                self.assertTrue(scipy.linalg.norm ( np.mean ( self.m_1.simulated_curves[cm][:,fwd,:], axis=1) -
                                                   self.m_1.simulated_curves[cm][0,fwd,0] ) < 1e-3 )

    def test_poly_eu_no_roots (self):
        """ Testing the roots part of _polynomial_european"
        """

        var_C = self.m_1.C_vec_list[0][11]  # TODO: CHANGE THIS
        K = 130.

        # CALL option testing 
        # C1 testing of the poly. european 
        for C1 in arange (-1., 1., 0.05):
            var_C_cur = array([C1, var_C[1], var_C[2]])
            res1 = self.m_1._polynomial_european(0, var_C_cur, 11, K, 1)
            res2 = self.m_1._polynomial_european(0, var_C_cur, 11, K, 1)
            self.assertLess( norm ( res1 - res2), 1e-6 )

        for C2 in arange (-2., 2., 0.05):
            var_C_cur = array([var_C[0], C2, var_C[2]])
            res1 = self.m_1._polynomial_european(0, var_C_cur, 11, K, 1)
            res2 = self.m_1._polynomial_european(0, var_C_cur, 11, K, 1)
            self.assertTrue( norm ( res1 - res2) < 1e-6 )

        for C3 in arange (-2., 4., 0.05):
            var_C_cur = array([var_C[0], var_C[1], C3])
            res1 = self.m_1._polynomial_european(0, var_C_cur, 11, K, 1)
            res2 = self.m_1._polynomial_european(0, var_C_cur, 11, K, 1)
            self.assertTrue ( norm ( res1 - res2) < 1e-6 )

        # PUT option testing
        K = 70.

        # C1 testing of the poly. european 
        for C1 in arange (-1., 1., 0.05):
            var_C_cur = array([C1, var_C[1], var_C[2]])
            res1 = self.m_1._polynomial_european(0, var_C_cur, 11, K, -1)
            res2 = self.m_1._polynomial_european(0, var_C_cur, 11, K, -1)
            self.assertLess( norm ( res1 - res2), 1e-6 )

        for C2 in arange (-2., 2., 0.05):
            var_C_cur = array([var_C[0], C2, var_C[2]])
            res1 = self.m_1._polynomial_european(0, var_C_cur, 11, K, -1)
            res2 = self.m_1._polynomial_european(0, var_C_cur, 11, K, -1)
            self.assertLess( norm ( res1 - res2), 1e-6 )

        for C3 in arange (-2., 4., 0.05):
            var_C_cur = array([var_C[0], var_C[1], C3])
            res1 = self.m_1._polynomial_european(0, var_C_cur, 11, K, -1)
            res2 = self.m_1._polynomial_european(0, var_C_cur, 11, K, -1)
            self.assertLess( norm ( res1 - res2), 1e-6 )

    def test_ln_model(self):

        sim_times = array([0.5, 1.])
        nb_sims = 10000000
        self.m_1.model_skew_ln_ind = 'ln_ln'
        self.m_1.update_sim_times (sim_times)
        self.m_1.simulate_curves(nb_sims)

        pr1 = pricers.call (self.m_1, [self.m_1.forward_curve_list[0][15],
                                      15, 1])

        disc_fact = self.m_1.DF (1.)
        theta = 1 # call options 
        iv = black_vol_inverse_vec (self.m_1.forward_curve_list[0][15],
                                         [self.m_1.forward_curve_list[0][15]],
                                         [pr1], 1., disc_fact, theta, 1e-4)

        self.assertTrue(abs (iv - self.m_1.atm_vol_list[0][15] ) < 1e-4)
