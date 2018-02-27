import config
import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import cublas
import cublas1
import unittest
import time
import scipy.linalg
from timing import time_cuda_call, time_normal_call


# THIS IS PERHAPS NOT NEEDED 
class cublasContext(ctypes.Structure):
    pass


class cublas_tests(unittest.TestCase):
    # reads the market object
    def setUp(self):
        print "Creating mtx A, x, y"
        size_pick = 'S'  # small or large matrix
        if size_pick == 'L':
            si1, si2, si3 = (12000, 80, 70)
        else:
            si1, si2, si3 = (120, 80, 70)
            
        self.A = np.random.normal(size=(si1, si2))
        self.A_gpu = gpuarray.to_gpu(self.A.astype(np.float32))
        self.A_sq = np.random.normal(size=(si1, si1))
        self.A_sq_gpu = gpuarray.to_gpu(self.A_sq.astype(np.float32))

        self.B = np.random.normal(size=(si2, si3))
        self.B_gpu = gpuarray.to_gpu(self.B.astype(np.float32))
        self.C = np.random.normal(size=(si1, si3))
        self.C_gpu = gpuarray.to_gpu(self.C.astype(np.float32))

        self.x = np.random.normal(size=(si2, 1))
        self.x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
        self.x1 = np.random.normal(size=(si2, 1))
        self.x1_gpu = gpuarray.to_gpu(self.x1.astype(np.float32))

        self.y = np.random.normal(size=(si1, 1))
        self.y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
        
        # for timing cuda 
        self.start = drv.Event()
        self.end = drv.Event()

    def tearDown(self):
        print "Destroying objects"
        self.A = None
        self.A_gpu = None
        self.A_sq = None
        self.A_sq_gpu = None
        self.B = None
        self.B_gpu = None
        self.C = None
        self.C_gpu = None
        self.x = None
        self.x_gpu = None
        self.x1 = None
        self.x1_gpu = None
        self.y = None
        self.y_gpu = None
        self.start = None
        self.end = None

    def test_Sdot_d(self):
        """
        # testing dot product
        """
        print "Testing the Sdot routine"
        y_res_gpu = cublas1.cublasSdot_d(self.x_gpu, self.x1_gpu)
        y_res_cpu = np.sum(self.x * self.x1)
        # print "y_gpu = ", y_res_gpu
        # print "y_cpu = ", y_res_cpu
        self.assertTrue(np.abs(y_res_gpu - y_res_cpu) < 0.1)

    # testing the quadratic form (A_d x_d, y_d)
    def test_quadf(self):
        # y_gpu' * A_sq * y_gpu 
        cpu_res = np.sum(np.dot(self.A_sq, self.y) * self.y)
        print "cpu res. =", cpu_res
        gpu_res = cublas.cublas_quadf(self.A_sq_gpu, self.y_gpu, self.y_gpu)
        print "gpu res. =", gpu_res
        diff1 = np.abs(cpu_res - gpu_res )
        print "diff     =", diff1
        self.assertTrue(diff1 < 0.2)  # THIS IS A REALLY LOOSE BOUND

    def test_sgemv_h(self):
        print "Testing the Sgemv routine"
        y_res = cublas.cublasSgemv_h(1., self.A, self.x, 0., self.y)
        y2 = 1. * np.dot(self.A, self.x)
        diff1 = np.sum((y2 - y_res)**2)
        self.assertTrue(diff1 < 5.)
    
    def test_sgemv_d(self):
        print "Testing the Sgemv routine"
        cublas.cublasSgemv_d (1., self.A_gpu, self.x_gpu, 1., self.y_gpu) # 
        y2 =  1. * np.dot ( self.A, self.x) + self.y
        diff1 = sum ( (y2 - self.y_gpu.get())**2 )
        print "diff1 = ", diff1

        self.assertTrue( diff1 < 1. )

    def test_sgemv_timing(self):
        # this first command transfers the data to the device, the second command 
        # is then much faster 
        cublas.cublasSgemv_d (2., self.A_gpu, self.x_gpu, 0., self.y_gpu)

        t3, irr = time_cuda_call(self.start, self.end, cublas.cublasSgemv_d, 
                                 2., self.A_gpu, self.x_gpu, 0., self.y_gpu )
        t1 = time.time()
        cublas.cublasSgemv_d (2., self.A_gpu, self.x_gpu, 0., self.y_gpu)
        print "finished gpu = ", time.time() - t1

        print "finished_gpu2 = ", t3

        t1 = time.time()
        x1 =  2. * np.dot (self.A, self.x) 
        print "finished cpu = ", time.time() - t1

        self.assertTrue (True)


    def test_sgemm(self):
        print "Testing the Sgemm routine"
        res_cublas = cublas.cublasSgemm_h (2., self.A, self.B, self.C) # 
        res_python = 2. * np.dot (self.A, self.B)

        diff1 = scipy.linalg.norm (res_cublas - res_python)
        print "difference = ", diff1

        self.assertTrue( diff1 <1. )

    def test_sgemm_timing (self):
        print "Testing the Sgemm routine"
        #cublas.cublasSgemm_h (2., self.A_gpu, self.B_gpu, 0., self.C_gpu) # 
        t1 = time.time()
        cublas.cublasSgemm_h (2., self.A, self.B, self.C) # 
        print "finished gpu = ", time.time() - t1

        t1 = time.time()
        C1 = 2. * np.dot (self.A, self.B)
        print "finished cpu = ", time.time() - t1

        self.assertTrue( True )


    # LU factorization testing 
    # def test_sgetrf_d (self):
    #     A_sq_magma_res = cublas.magma_sgetrf_d ( self.A_sq_gpu )
    #     A_sq_lu_res = scipy.linalg.lu_factor( self.A_sq)[0]

    #     diff1 = scipy.linalg.norm (A_sq_magma_res - A_sq_lu_res )
    #     print "diff = ", diff1

    #     self.assertTrue (diff1 < 1.)

    def test_sgetrf_h (self):
        A_sq_magma_res = cublas.magma_sgetrf_h ( self.A_sq )
        A_sq_lu_res = scipy.linalg.lu_factor( self.A_sq)[0]

        diff1 = scipy.linalg.norm (A_sq_magma_res - A_sq_lu_res )
        print "diff = ", diff1

        self.assertTrue (diff1 < 1.)

    # testing qr factorization
    # r matrix obtained correctly, q _not_ finalized !!
    def test_sgeqrf_h (self):
        t1 = time.time()
        A_sq_magma_res = cublas.magma_sgeqrf_h ( self.A_sq )
        t_gpu = time.time() - t1
        #print A_sq_magma_res
        t1 = time.time()
        A_sq_qr_res = np.linalg.qr (self.A_sq)
        t_cpu = time.time() -t1
        #print A_sq_qr_res
        
        print "gpu time = ", t_gpu
        print "cpu time = ", t_cpu

        #diff1 = scipy.linalg.norm (A_sq_magma_res - A_sq_lu_res )
        #print "diff = ", diff1

        self.assertTrue (True)


    # General linear system testing 
    # THIS HAS A LARGE ERROR, INVESTIGATE
    def test_sgesv(self):
        # A _HAS_ to be a square matrix 
        si = 400
        A = np.random.random_sample((si, si))
        b = np.random.random_sample((si,150))

        t1 = time.time()
        sol1 = cublas.magma_sgesv_h(A, b)
        t_gpu = time.time() - t1

        t1 = time.time()
        sol2 = np.linalg.solve (A, b)
        t_cpu = time.time() - t1

        print "gpu time = ", t_gpu
        print "cpu time = ", t_cpu
        
        # difference between the solutions 
        diff1 = scipy.linalg.norm(sol1 - sol2 )
        print "diff = ", diff1

        self.assertTrue (diff1 < 1.)



# running the tests 
suite = unittest.TestLoader().loadTestsFromTestCase(cublas_tests)
unittest.TextTestRunner(verbosity=2).run(suite)
