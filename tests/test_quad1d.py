import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import unittest
import time

import scipy.linalg
from scipy.optimize import brenth, fmin
import multiprocessing as mp
import mrds.quad1d as quad1d


# main testing class 
class Quad1DTests(unittest.TestCase):

    # TODO: REDO THESE TESTS.
    # reads the market object 
    def setUp (self):
        si1 = 120
        si2 = 80
        si3 = 70
        self.A = np.random.random_sample((si1,si2)).astype(np.float32)
        self.A_gpu = gpuarray.to_gpu (self.A)
        self.A_sq = np.random.random_sample((si1,si1))
        self.A_sq_gpu = gpuarray.to_gpu (self.A_sq)

        self.B = np.random.random_sample((si2,si3)).astype(np.float32)
        self.B_gpu = gpuarray.to_gpu (self.B)
        self.C = np.random.random_sample((si1,si3)).astype(np.float32)
        self.C_gpu = gpuarray.to_gpu (self.C)

        self.x = np.random.random_sample(si2).astype(np.float32)
        self.x_gpu = gpuarray.to_gpu (self.x)
        self.y = np.random.random_sample(si1).astype(np.float32)
        self.y_gpu = gpuarray.to_gpu (self.y)


    # testing gauss-hermite quadrature 
    def test_gh_quad_gpu (self):
        """ Testing the gauss-hermite quadrature on GPU (1 dim)"""

        params = np.arange(1., 30., 0.05)
        res = []
        f = lambda x, p: max (p * np.exp(x) - 1., 0.)

        x,w = quad1d.gh_pw_hints(100) # this _must_ be 100

        t1 = time.time()
        for p in params:
            res.append ( quad1d.gh_quad_for_test(x,w,lambda x: f(x,p)) )
        res = np.array(res)
        t_cpu1 = time.time() - t1

        t1 = time.time()
        res2 = quad1d.gh_quad_for_test_2(x,w,params )
        t_cpu2 = time.time() - t1

        t1 = time.time()
        res3 = quad1d.gh_quad_spread_gpu (x,w, params)
        t_gpu = time.time() - t1

        diff1 = scipy.linalg.norm (res - res2 )
        diff2 = scipy.linalg.norm (res2 - res3 )

        self.assertLess( diff1, 0.5 )
        self.assertLess(diff2, 0.5)

    def test_gh_impl_quad_gpu (self):
        """Testing the gauss-hermite quadrature search on GPU (1 dim)
        """

        target = 5.
        x,w = quad1d.gh_pw_hints(100) # this _must_ be 100

        # CPU part
        f_o = lambda p: sum ( w * np.maximum (p * np.exp(x) - 1., 0) ) - target # optimizing function 
        t1 = time.time()
        res1 = brenth (f_o, 0.01, 50.)
        t_cpu1 = time.time() - t1

        # GPU part
        t1 = time.time()
        res2 = quad1d.gh_impl_quad_spread_gpu (x, w, target)
        t_gpu = time.time() - t1

        diff1 = scipy.linalg.norm (res1 - res2 )

        self.assertTrue( diff1 < 0.5 ) 




    # testing gauss-hermite quadrature 
    def test_gh_quad_2d_gpu (self):
        """Testing the gauss-hermite quadrature on GPU (2 dim)"""

        target = 6.
        pr_size = 100
        x,w = quad1d.gh_pw_hints(pr_size)

        ones_col = np.exp (x.reshape((pr_size,1)) )
        ones_row = np.exp (x)
        w_row = w
        w_col = w.reshape((pr_size,1))
        f_o = lambda p : (np.sum (
                np.kron (w_col, w_row) *
                np.maximum ( p[0] * np.kron (ones_col, np.ones(pr_size) ) + 
                             p[1] * np.kron ( np.ones(pr_size).reshape((pr_size,1)), ones_row ) - 1., 0.) 
                ) - target)**2
        t1 = time.time()
        res_cpu = fmin (f_o, np.array([1., 1.]))
        t_cpu = time.time() - t1

        t1 = time.time()
        res_gpu = quad1d.gh_quad_2d_trispread_gpu (x,w, target)
        t_gpu = time.time() - t1

        self.assertTrue( True)



    # testing gauss-hermite quadrature 
    def test_cva_gh_quad_2d_gpu (self):
        """Testing the gauss-hermite quadrature on GPU (2 dim)"""

        pr_size = 100
        x,w = quad1d.gh_pw_hints(pr_size)

        ones_col = np.exp (x.reshape((pr_size,1)) )
        ones_row = np.exp (x)
        w_row = w
        w_col = w.reshape((pr_size,1))
        w_col_row = np.kron (w_col, w_row)
        p0_mat = np.kron (ones_col, np.ones(pr_size) )
        p1_mat = np.kron ( np.ones(pr_size).reshape((pr_size,1)), ones_row )
        f_o = lambda p : np.sum (
                w_col_row *
                np.maximum ( p[0] * p0_mat + 
                             p[1] * p1_mat - 1., 0.) 
                )

        p_1 = np.arange(0.1, 100., 0.1)
        p_2 = np.arange(0.1, 10., 0.1)
        res_cpu = np.zeros ( (len(p_1), len(p_2) ) )

        # above 2 loops with multiprocessing:
        t1 = time.time()
        nb_cores = mp.cpu_count() 
        pool = mp.Pool(processes=nb_cores)
        p1_l = len(p_1)
        res_mp = pool.map ( quad1d.gh_2d_integr_cpu, 
                            zip ( p_1, [p_2] * p1_l, [x] * p1_l, [w] * p1_l, 
                                  [pr_size] * p1_l ) 
                            )
        t_cpu_mt = time.time() - t1


        t1 = time.time()
        for p1_ind,p1 in enumerate(p_1):
            for p2_ind,p2 in enumerate(p_2):
                res_cpu[p1_ind, p2_ind] = f_o( np.array([p2,p1]))
        t_cpu = time.time() - t1



        # GPU processing 
        t1 = time.time()
        res_gpu = quad1d.gh_cva_quad_2d_trispread_gpu ( x, w, p_1, p_2 ) 
        t_gpu = time.time() - t1

        self.assertTrue( True)



# running the tests 
def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(Quad1DTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
