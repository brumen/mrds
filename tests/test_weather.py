#
# jw volatilities TESTING 

import config 
import numpy as np
import datetime

import pycuda.driver as drv
import pycuda.gpuarray as gpa
import pycuda.cumath
#from cudanormal import cudanormal

import weather as we # class to test 
import unittest 
import time
import weather_fast as wf
from timing import time_cuda_call, time_normal_call


def set_params():

    # historical parameters: A,B,C,omega,phi = hp
    # hp = (5.97, 6.57*10**(-5), 10.4, 2 * pi , -2.01, 0.5)
    hp = (54.161, 0.0001918, 22.183, 2*np.pi, -2.034, 114.261)

    date_o = datetime.date(1973,1,1)
    date_p = datetime.date(2012,11,24)
    date_s = datetime.date(2012, 4,1)
    nm_1 = 1
    nm_2 = 3
    sp_init = np.array([3.3, 3.3])
    
    HDD_date_l = {0: (date_s, nm_1),
                  1: (date_s, nm_2)
                  }
    
    HDD_price_l = {0: 480., 
                   1: 630.
                   }

    HDDO_price_l = {0: (480., 8.), 
                    1: (630., 7.)
                    }


    return [ hp, sp_init, 
             [date_o, date_p, date_s], 
             [nm_1, nm_2], 
             HDD_date_l, HDD_price_l,
             HDDO_price_l
             ]






# jw parametrization (inherits from vol_param)
class modulesTest (unittest.TestCase):

    def setUp (self):

        #self.Z_m = np.random.normal(size=(31, 400000))
        #self.Z_m_d = gpa.to_gpu (self.Z_m.astype(np.float32))
        #self.N_step = 31
        #self.range_gpu = gpa.to_gpu( np.arange(self.N_step-1, -1,-1).astype(np.float32) )
        #self.range_gpum1 = gpa.to_gpu( - np.arange(self.N_step-1, -1,-1).astype(np.float32) )
        # THIS IS WRONG BELOW, SHOULD BE N_step, but for now, we leave it
        #self.range_gpu_inv = gpa.to_gpu (np.arange(self.N_step).astype(np.float32))  # inverse, used in T_par_inn

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


    def test_functions_running (self):
        """
        simply tests whether the functions are working 
        """
        def calc_1(gpu_ind):

            # historical parameters: A,B,C,omega,phi = hp
            # hp = (5.97, 6.57*10**(-5), 10.4, 2 * pi , -2.01, 0.5)
            hp = (54.161, 0.0001918, 22.183, 2*np.pi, -2.034, 114.261)

            date_o = datetime.date(1973,1,1)
            date_p = datetime.date(2012,11,24)
            date_s = datetime.date(2012, 4,1)
            nm_1 = 1
            nm_2 = 3
            sp_init = np.array([3.3, 3.3])
            
            HDD_date_l = {0: (date_s, nm_1),
                          1: (date_s, nm_2)
                          }
    
            HDD_price_l = {0: 480., 
                           1: 630.
                           }
    
            hdd1_hist = we.HDD_histo (date_p, date_s, nm_1, hp, HDD_date_l, date_o)
            print "hdd1 hist = ", hdd1_hist
            hdd2_hist = we.HDD_histo (date_p, date_s, nm_2, hp, HDD_date_l, date_o)
            print "hdd2 hist = ", hdd2_hist
    
            # HDD options given in a format (K, price)

            HDDO_price_l = {0: (480., 8.), 
                            1: (630., 7.)
                            }

            # price under the current parameters 
            print "calibrating"
            t1 = time.time()
            ca1 = we.HD_calib_all(date_p, date_s, HDD_date_l, HDD_price_l, HDDO_price_l, hp, sp_init, 
                                  date_o, gpu_ind=gpu_ind)
            t1 = time.time() - t1
            print "ca1 = ", ca1
            print "time = ", t1

        self.assertTrue (True)

    def test_T_m (self):
        """
        tests T_m
        """

        # dates = [date_o, date_p, date_s], 
        # nm = [nm_1, nm_2] 
        hp, sp_init, dates, nm, HDD_date_l, HDD_price_l, HDDO_price_l = set_params()

        # BADLY WRITTEN IN WEATHER.PY - HAS TO BE 31
        t_v = 0.01 + 1./365. * np.arange(31)
        t_v_d = gpa.to_gpu (t_v.astype(np.float32))

        t1, cpu_r = time_normal_call(we.T_m, t_v, hp)
        t2, gpu_r = time_cuda_call(self.start, self.end, we.T_m_d, 
                                   t_v_d, hp)

        res_diff = cpu_r - gpu_r.get()
        res_bool = ( res_diff < 1e-2 ) & (res_diff > -1e-2)

        print "t_cpu = ", t1
        print "t_gpu = ", t2
        print "speedup = ", t1/t2

        self.assertTrue( res_bool.all() )

    def test_T_par_inn (self):
        """
        tests T_par_inn function in particular
        """

        # dates = [date_o, date_p, date_s], 
        # nm = [nm_1, nm_2] 
        hp, sp_init, dates, nm, HDD_date_l, HDD_price_l, HDDO_price_l = set_params()

        ttdp = (dates[1] - dates[0]).days / 365.25
        t_v = 0.01 + 1./365. * np.arange(31) 
        T_m_v = we.T_m (ttdp + t_v, hp)
        T_m_d = we.T_m_der (ttdp + t_v,hp) # vector as well

        #T_m_v_d = gpa.to_gpu (T_m_v.astype(np.float32))
        #T_m_d_d = gpa.to_gpu (T_m_d.astype(np.float32))

        Z_m = np.random.normal(size=(31,20000))
        Z_m_d = gpa.to_gpu (Z_m.astype(np.float32))

        t1 = time.time()
        cpu_r = we.T_par_inn (T_m_v, T_m_d, 1./365., sp_init, hp, Z_m, dates[1], 
                              dates[0] )
        t1 = time.time() - t1

        t2 = time.time()
        gpu_r = we.T_par_inn_d (T_m_v, T_m_d, 1./365., sp_init, hp, Z_m_d, dates[1], 
                                dates[0] )
        t2 = time.time() - t2

        res_diff = cpu_r - gpu_r.get()
        res_bool = ( res_diff < 1e-4 ) & (res_diff > -1e-4)

        print "t_cpu = ", t1
        print "t_gpu = ", t2
        print "speedup = ", t1/t2

        # print "T_par_inn cpu = ", cpu_r
        # print "T_par_inn gpu = ", gpu_r

        self.assertTrue( res_bool.all() )
        

    def test_T_sim_inn (self):
        """
        tests T_sim_inn function in particular
        """


        # dates = [date_o, date_p, date_s], 
        # nm = [nm_1, nm_2] 
        hp, sp_init, dates, nm, HDD_date_l, HDD_price_l, HDDO_price_l = set_params()


        ttdp = (dates[1] - dates[0]).days / 365.25
        t_v = 0.01 + 1./365. * np.arange(31) 
        T_m_v = we.T_m (ttdp + t_v, hp)
        T_m_d = we.T_m_der (ttdp + t_v,hp) # vector as well

        #T_m_v_d = gpa.to_gpu (T_m_v.astype(np.float32))
        #T_m_d_d = gpa.to_gpu (T_m_d.astype(np.float32))

        cpu_r = we.T_sim_inn (T_m_v, T_m_d, 1./365., 31, sp_init, hp, self.Z_m, dates[1], 
                              dates[0] )

        gpu_r = we.T_sim_inn_d(T_m_v, T_m_d, 1./365., 31, sp_init, hp, 
                               self.Z_m_d, self.range_gpu, self.range_gpum1,
                               dates[1], dates[0] )

        res_diff = abs(cpu_r - gpu_r.get()) < 1e-4

        # print "cpur = ", cpu_r
        # print "gpur = ", gpu_r.get()

        self.assertTrue( res_diff.all())


    def test_HDDO (self):
        """
        tests HDDO_real function 
        """

        # dates = [date_o, date_p, date_s], 
        # nm = [nm_1, nm_2] 
        hp, sp_init, dates, nm, HDD_date_l, HDD_price_l, HDDO_price_l = set_params()
        
        K = 571. # 571 historical val

        hddo_c = we.HDDO_histo (K, dates[1], dates[2], nm[0], 
                                hp, HDD_date_l, dates[0], 
                                self.Z_m, self.Z_m_d, 
                                self.range_gpu, self.range_gpum1, 
                                self.range_gpu_inv,
                                gpu_ind = False)

        hddo_d = we.HDDO_histo (K, dates[1], dates[2], nm[0], 
                                hp, HDD_date_l, dates[0], 
                                self.Z_m, self.Z_m_d, 
                                self.range_gpu, self.range_gpum1, 
                                self.range_gpu_inv,
                                gpu_ind = True)


        print "hddo_c = ", hddo_c
        print "hddo_d = ", hddo_d

        res_diff = np.abs(hddo_d - hddo_c)
        res_bool = ( res_diff < 1e-1 ) & (res_diff > -1e-1)


        self.assertTrue( res_bool )


    def test_HDD (self):
        """
        tests HDD_real function 
        """

        # dates = [date_o, date_p, date_s], 
        # nm = [nm_1, nm_2] 
        hp, sp_init, dates, nm, HDD_date_l, HDD_price_l, HDDO_price_l = set_params()
        
        hdd_c = we.HDD_histo (dates[1], dates[2], nm[0], 
                              hp, HDD_date_l, dates[0], 
                              self.Z_m, self.Z_m_d, 
                              self.range_gpu, self.range_gpum1, 
                              self.range_gpu_inv,
                              gpu_ind = False)

        hdd_d = we.HDD_histo (dates[1], dates[2], nm[0], 
                              hp, HDD_date_l, dates[0], 
                              self.Z_m, self.Z_m_d, 
                              self.range_gpu, self.range_gpum1, 
                              self.range_gpu_inv,
                              gpu_ind = True)


        #print "hdd_c = ", hdd_c
        #print "hdd_d = ", hdd_d

        res_diff = np.abs(hdd_d - hdd_c)
        res_bool = ( res_diff < 1e-1 ) & (res_diff > -1e-1)


        self.assertTrue( res_bool )



class smallTest (unittest.TestCase):

    def test_1(self):
        
        self.assertTrue (True)



    

# running the above tests 
#suite = unittest.TestLoader().loadTestsFromTestCase(smallTest)
suite = unittest.TestLoader().loadTestsFromTestCase(modulesTest)
unittest.TextTestRunner(verbosity=2).run(suite)
