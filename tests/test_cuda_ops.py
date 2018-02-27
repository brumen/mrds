#
# testing of cuda operations
#
import config 
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpa
import pycuda.cumath
import unittest
import time
from timing import time_cuda_call, time_normal_call
import cuda_ops as co 


def test_vpm_cols(pt_ind='p',
                  rows_l=[31, 300, 3000, 10000],
                  cols_l=[40, 400, 4000, 10000]):
    """
    testing the vector + matrix routine
    RESULT: only works on matrices with many columns
    """
    def execute_vpm(rows, cols):
        v1 = np.random.random(rows)
        v1r = v1.reshape((rows, 1))
        m1 = np.random.random((rows, cols))
        m1_d = gpa.to_gpu(m1).astype(np.float32)
        t1, res_cpu = time_normal_call(lambda : v1r + m1)
        t2, irr = time_cuda_call(drv.Event(), drv.Event(), co.vtpm_cols, v1, m1_d, pt_ind)
        res_gpu = m1_d.get()
        res_diff = np.abs(res_cpu - res_gpu)
        res_bool = (res_diff < 1e-4)
        print "rows, cols: ", (rows, cols), "gpu speedup :", t1/t2

    for rows_nb in rows_l:
        for cols_nb in cols_l:
            execute_vpm(rows_nb, cols_nb)


def test_vtpv(pt_ind='t',
              rows_l=[5, 300, 3000, 10000, 50000, 500000],
              cols_l=[63]):
    """
    testing the vector + vector that makes a matrix
    """
    def execute_vpm(rows, cols):
        v1 = np.random.random(rows).astype(np.float32)
        v1_d = gpa.to_gpu(v1)
        m1 = np.random.random(cols).astype(np.float32)
        m1_d = gpa.to_gpu(m1)
        t1, res_cpu = time_normal_call(lambda: v1.reshape((rows, 1)) * m1)
        t2, res_gpu = time_cuda_call(drv.Event(), drv.Event(), co.vtpv, v1_d, m1_d, 't')
        # for i in range(10000):
        #     res_gpu = co.vtpv(v1_d, m1_d, 't')
        # t2 = 1.
        res_gpu = res_gpu.get()
        # print "V1", v1_d, m1_d
        # print "RES_GPU", res_gpu, "\n", res_cpu
        res_diff = np.abs(res_cpu - res_gpu)
        print "rows, cols: ", (rows, cols), "gpu speedup :", t1 / t2, "diff:", np.max(res_diff)

    for rows_nb in rows_l:
        for cols_nb in cols_l:
            execute_vpm(rows_nb, cols_nb)

    v1 = np.random.random(10000).astype(np.float32)
    v1_d = gpa.to_gpu(v1)
    m1 = np.random.random(63).astype(np.float32)
    m1_d = gpa.to_gpu(m1)

    co.vtpv(v1_d, m1_d, tm_ind='t')

    # for idx in range(100000):
    #    print "IDX", idx
    #    co.vtpv(v1_d, m1_d, 't')


# same as above, just tests the multiplication
def test_rowsum_cuda(rows_l=[31, 300, 3000, 10000],
                     cols_l=[40, 400, 4000, 10000]):

    def execute_rowsum(row_nb, col_nb):
        m1 = np.random.random((row_nb, col_nb))
        m1_d = gpa.to_gpu(m1).astype(np.float32)
        t1, res_cpu = time_normal_call(np.sum, m1, axis=1)
        ones_d = gpa.to_gpu(np.ones((31, 1)).astype(np.float32))
        res_d = gpa.empty((row_nb, 1), dtype=np.float32)
        t2, res_gpu_d = time_normal_call(co.rowsum_cuda, m1_d, ones_d, res_d)
        res_gpu = res_cpu  # res_gpu_d.get()
        res_diff = res_cpu - res_gpu
        res_bool = np.abs(res_diff) < 1e-2
        print "row, nb, sppedup:", (row_nb, col_nb), t1/t2
        return res_bool.all()

    for rows_nb in rows_l:
        for cols_nb in cols_l:
            execute_rowsum(rows_nb, cols_nb)


class modulesTest (unittest.TestCase):
    def setUp(self):
        # for timing cuda
        self.start = drv.Event()
        self.end = drv.Event()
        self.large_small_ind = 'L'
        self.rows_large = 31
        self.cols_large = 400000
        self.rows_small = 50
        self.cols_small = 310

        if self.large_small_ind == 'L':
            self.rows = self.rows_large
            self.cols = self.cols_large
        else:
            self.rows = self.rows_small
            self.cols = self.cols_small

        print "Matrix size = (", self.rows, ",", self.cols, ")"

    def test_colsum_cuda(self):

        sample_size_rows = 31
        sample_size_cols = 400000

        m1 = np.random.random((sample_size_rows, sample_size_cols))
        m1_d = gpa.to_gpu(m1).astype(np.float32)

        start = drv.Event()
        end = drv.Event()

        t1, res_cpu = time_normal_call(np.cumsum, m1, axis=0)
        t2, irr = time_cuda_call(start, end, co.colsum_cuda, m1_d)
        res_gpu = m1_d.get() 
        res_bool = np.abs(res_cpu - res_gpu) < 1e-2

        # print "rcpu = ", res_cpu
        # print "rgpu = ", res_gpu
        print "time cpu = ", t1
        print "time gpu = ", t2
        print "speedup = ", t1/t2
        self.assertTrue(res_bool.all())

    def test_colsum_cuda_last(self):

        sample_size_rows = 31
        sample_size_cols = 400000

        m1 = np.random.random((sample_size_rows, sample_size_cols))
        v1 = np.zeros(sample_size_cols)
        m1_d = gpa.to_gpu(m1).astype(np.float32)
        # v1_d = gpa.zeros(sample_size_cols, np.float32)
        res_d = gpa.to_gpu(v1).astype(np.float32)

        start = drv.Event()
        end = drv.Event()

        t1, res_cpu = time_normal_call(np.sum, m1, axis=0)
        t2, res_d = time_cuda_call(start, end, co.colsum_cuda_last, m1_d)
        res_gpu = res_d.get() 
        res_bool = np.abs(res_cpu - res_gpu) < 1e-2
        print "rcpu = ", res_cpu
        print "rgpu = ", res_gpu
        print "time cpu = ", t1
        print "time gpu = ", t2
        print "speedup = ", t1/t2

        self.assertTrue(res_bool.all())

    def test_rowsum_cuda_notransfer(self):

        sample_size_rows = self.rows
        sample_size_cols = self.cols 

        m1 = np.random.random((sample_size_rows, sample_size_cols))
        m1_d = gpa.to_gpu(m1).astype(np.float32)
        v1_d = gpa.zeros(sample_size_rows, np.float32)

        t1, res_cpu = time_normal_call(np.sum, m1, axis=1)
        t2, v1_d = time_cuda_call(self.start, self.end, co.rowsum_cuda_notransfer, 
                                  m1_d, v1_d)
        res_gpu = v1_d.get()
        res_diff = res_cpu - res_gpu
        res_bool = np.abs(res_diff) < 1e-2

        print "time cpu = ", t1
        print "time gpu = ", t2
        print "speedup = ", t1/t2
        
        self.assertTrue(res_bool.all())

    def test_maximum_cuda(self):
        """
        same as above, just tests the multiplication
        """
        sample_size_rows = self.rows
        sample_size_cols = self.cols

        m1 = np.random.random((sample_size_rows, sample_size_cols)) - 0.5
        m1_d = gpa.to_gpu(m1).astype(np.float32)

        t1, res_cpu = time_normal_call(np.maximum, m1, 0.)
        t2, irr = time_cuda_call(self.start, self.end, co.maximum_cuda, m1_d)
        res_gpu = m1_d.get()

        res_diff = res_cpu - res_gpu
        res_bool = (res_diff < 1e-2) & (res_diff > -1e-2)

        print "time cpu = ", t1
        print "time gpu = ", t2
        print "speedup = ", t1/t2
        
        self.assertTrue(res_bool.all())

    def test_write_vec_to_mat_col(self):
        """
        copies v1 on m1 in the 2nd column, 
        """
        sample_size_rows = self.rows

        v1 = np.random.random(sample_size_rows) - 0.5
        v1_d = gpa.to_gpu(v1).astype(np.float32)
        m1 = np.zeros((sample_size_rows,5))
        m1_d = gpa.to_gpu(m1).astype(np.float32)

        t2 = time.time()
        co.write_vec_in_mat_col (v1_d, m1_d, 1)
        t2 = time.time() - t2

        res_gpu = m1_d.get()[:, 1]
        res_diff = res_gpu - v1
        res_bool = (res_diff < 1e-2) & (res_diff > -1e-2)

        self.assertTrue(res_bool.all())

    def test_sin_cos(self):
        """
        tests sin_cos_exp_d function 
        """
        t_v = 0.01 + 7./36. * np.arange(500000)
        t_v_d = gpa.to_gpu(t_v.astype(np.float32))
        t_v_d1 = gpa.to_gpu(t_v.astype(np.float32))

        t1, cpu_r = time_normal_call(np.sin, t_v)
        t2, irr = time_cuda_call(self.start, self.end, co.sin_cos_exp_d, 
                                 t_v_d, t_v_d1, 'sin')
        t3, gpu_r = time_cuda_call(self.start, self.end, pycuda.cumath.sin,
                                   t_v_d)

        res_diff = cpu_r - t_v_d1.get()
        res_bool = (res_diff < 1e-2) & (res_diff > -1e-2)

        print "t_cpu = ", t1
        print "t_gpu = ", t2
        print "t_gpu_2 = ", t3
        print "speedup = ", t1/t2

        self.assertTrue( res_bool.all())

    def test_exp_fct(self):
        """
        tests sin_cos_exp_d function for EXP ONLY 
        """
        t_v = 0.01 + 1e-4 * np.arange(500000)
        t_v_d = gpa.to_gpu(t_v.astype(np.float32))
        t_v_d1 = gpa.to_gpu(t_v.astype(np.float32))

        t1, cpu_r = time_normal_call(np.exp, t_v)
        t2, irr = time_cuda_call(self.start, self.end, co.sin_cos_exp_d, 
                                 t_v_d, t_v_d1, 'exp')
        t3, gpu_r = time_cuda_call(self.start, self.end, pycuda.cumath.exp,
                                   t_v_d)

        res_diff = cpu_r - t_v_d1.get()
        res_bool = (res_diff < 1e-2) & (res_diff > -1e-2)

        print "t_cpu = ", t1
        print "t_gpu (Mine) = ", t2
        print "t_gpu (Builtin) = ", t3
        print "speedup (Mine) = ", t1/t2
        print "speedup (Builtin) = ", t1/t3

        self.assertTrue(res_bool.all())


def test_matmul():
    """
    tests the matrix multiplication
    """
    g1 = gpa.zeros((500, 200), dtype=np.float32) + 1.
    g2 = gpa.zeros((200, 500), dtype=np.float32) + 1.
    g3 = gpa.zeros((500, 500), dtype=np.float32)
    for idx in range(10000):
        co.matmul(g1, g2, g3)
    return g3[100, 100] == 200.


def test_vtpm_ao(pt_ind='t',
                 rows_l=[5, 300, 3000, 10000, 50000, 500000],
                 cols_l=[50, 70, 100]):
    """
    testing vtpm for ao
    """
    def execute_vpm(rows, cols):
        v_plus_h = np.random.random((rows, 1))
        v_plus_d = gpa.to_gpu(v_plus_h)
        v_mult_h = np.random.random((rows, 1))
        v_mult_d = gpa.to_gpu(v_mult_h)
        m_h = np.random.random((rows, cols))
        m_d = gpa.to_gpu(m_h)
        # t1, res_cpu = time_normal_call(lambda: v1.reshape((rows, 1)) * m1)
        # t2, res_gpu = time_cuda_call(drv.Event(), drv.Event(), co.vtpv, v1_d, m1_d, 't')
        print "start cpu"
        m_h *= v_mult_h
        m_h += v_plus_h
        print "finish cpu"
        print "start gpu"
        res_gpu = co.vtpm_cols_new_hd_ao(v_plus_h, v_mult_h, m_d)
        print "finish gpu"
        res_gpu = res_gpu.get()
        res_diff = np.abs(m_h - res_gpu)
        print "RR", rows, cols, np.max(res_diff)
        # print "rows, cols: ", (rows, cols), "gpu speedup :", t1 / t2, "diff:", np.max(res_diff)

    for rows_nb in rows_l:
        for cols_nb in cols_l:
            execute_vpm(rows_nb, cols_nb)


class smallTest (unittest.TestCase):
    """
    conducts small range tests
    """
    def test_1(self):
        self.assertTrue(True)


# running the above tests 
# suite = unittest.TestLoader().loadTestsFromTestCase(smallTest)
# suite = unittest.TestLoader().loadTestsFromTestCase(modulesTest)
# unittest.TextTestRunner(verbosity=2).run(suite)

# a1 = 5. * np.ones((500,1000)).astype(np.float32)
# a1_d = gpa.to_gpu(a1)
# co.cumsum_cuda(a1_d)
# print a1_d
# test_vtpv()
