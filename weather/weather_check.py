#
# weather model, as described in 


import numpy as np
import openopt # optimization solver
import pycuda.curandom
import pycuda.gpuarray as gpa
import pycuda.cumath
import pycuda.reduction
from pycuda.compiler import SourceModule
#from cudanormal import cudanormal
import datetime
import calendar

import cublas
from cuda_ops import *


class weather():
    def __init__(self, sim_size, hp, date_l, sp=np.array([3.3, 3.3]),
                 gpu_ind = False):
        """
        sim_size ... simulation size 
        hp ... historical parameters 
        sp ... simulation parameters
        date_l ... date list of [date_o, date_p, date_s]
        """
        self.sim_size = sim_size
        self.N_step = 31
        self.gpu_ind = gpu_ind # True for CUDA, False for CPU 

        self.hp = hp # historical parameters 
        self.sp = sp # simulation parameters 
        self.date_o, self.date_p, self.date_s = date_l # origin, pricing, start date 

        self.T_help_d = gpa.zeros(self.N_step, np.float32)
        self.T_d_help_d = gpa.zeros(self.N_step, np.float32)
        
        self.Z_m = None
        self.Z_m_d = None 
        self.update_sim_nb(sim_size) # constuct Z_m, Z_m_d 

    def update_sim_nb(self, sim_size):
        self.Z_m = np.random.normal(size=(31, sim_size))
        self.Z_m_d = gpa.to_gpu (self.Z_m.astype(np.float32))

    # average function
    def T_m (self, t):
        A, B, C, omega, phi, a = self.hp # a is the mean reversion params
        return A + B * t + C * np.sin (omega * t + phi)

    # average function for t_d on the device
    def T_m_d (self, t_d):
        A, B, C, omega, phi, a = self.hp # a is the mean reversion params
        t_d1 = t_d *omega + phi
        sin_cos_d(t_d1, self.T_help_d)
        return A + B * t_d + C * self.T_help_d

    def T_m_real (self, t_date):
        t_num = (t_date - self.date_o).days / 365.25
        return self.T_m (t_num)

    def T_m_der (self, t):
        A, B, C, omega, phi, a = self.hp
        return B + C * np.cos (omega * t + phi) * omega

    # same as above function, it just does the t_d on the device
    def T_m_der_d (self, t_d):
        A, B, C, omega, phi, a = self.hp
        # implements: return B + C * pycuda.cumath.cos (omega * t_d + phi) * omega
        sin_cos_d(omega * t_d + phi, self.T_d_help_d, 'cos')
        return B + ( C * omega ) * self.T_d_help_d


    def T_m_der_real (self, t_date, t_origin):
        t_num = (t_date - t_origin).days / 365.25
        return self.T_m_der (t_num)


    def T_step (self, t, dt, T_t_v, Z):
        """
        step of simulation 
          sp ... simulation parameters (sigma, lam) 
          hp ... historical parameters, see T_m
          Z is a vector of same len. as T_t_v
        """

        T_m_v = self.T_m (t)
        T_m_d = self.T_m_der (t)
        a = self.hp[5]
        sigma, lam = self.sp
        return T_t_v + (T_m_d + a * (T_m_v - T_t_v ) - lam * sigma ) * dt + sigma * np.sqrt (dt) * Z

    # complete simulates the whole process, slow 
    def T_sim (self, t_0, t_step, T_0 ):
        T_0_v = T_0 * np.ones(self.Z_m.shape[0])
        T_s = np.zeros ((self.Z_m.shape[0], self.Z_m.shape[1]+1)) # simulated matrix (
        T_s[:,0] = T_0_v
        for n in np.arange(1,self.N_step+1):
            T_s[:,n] = self.T_step(t_step * n, t_step, T_s[:,n-1], self.Z_m[:,n-1])
    
        return T_s

    def HDD (self, t_0, t_step, T_0, sp):
        ttdp = (self.date_p - self.date_o).days / 365.25
        t_v = t_0 + t_step * np.arange(self.N_step) 
        T_m_v = self.T_m (ttdp + t_v )
        T_m_d1 = self.T_m_der (ttdp + t_v) # vector as well

        T_sm = self.T_sim_inn ( T_m_v, T_m_d1, t_step, sp ) # T simulated matrix 

        return self.HDD_payoff(T_sm)


    def month_into_sigma(self, n, mi_dec):
        """
        maps month n = 0, 1, 2, 3 into sp_l index when mi_dec is given 
        """
        np1 = n+1
        return sum ([ m[1]<=np1 for m in mi_dec.values() ])

    def HDD_real (self, nb_months, sp_l, HDD_date_l ):
        """
        incorporates correct handling of dates 
        sp_l, hp_l are lists of simulation parameters, historical parameters for months 
        date_p ... pricing date (datetime format)
        date_s ... start date, 
        date_o ... origin date (check what that really is)
        nb_sims. months ... number of months of the HDD
        """

        mi = self.months_index (HDD_date_l, sp_l[0]) # just need month_decom 
        mi_dec = self.month_decomp (mi["month_decomp"])

        hdd_val = 0.
        for n in np.arange (nb_months):
            t_month_start = self.add_months (self.date_s, n) 
            t_month_start_num = (t_month_start - self.date_p).days / 365.
        
            T_0 = self.T_m_real (t_month_start)
            t_step = 1. / 365.
            # N_step = (self.add_months(t_month_start, 1) - t_month_start).days 
            # CHECK THIS MIGHT BE WRONG 
            sp = sp_l[self.month_into_sigma(n,mi_dec)] # mapping into sp_l[ n
            if self.gpu_ind == False:
                hdd_val += self.HDD( t_month_start_num, t_step, T_0, sp)
            else:
                # 31 IS WRONG, due to impossible slicing in gpuarray
                hdd_val += self.HDD_d( t_month_start_num, t_step, T_0, sp )

        return hdd_val

    def HDD_histo (self, nb_months, HDD_date_l):

        sp_l = [(0.,0.)] * len(HDD_date_l)
        return self.HDD_real ( nb_months, sp_l, HDD_date_l )


    def month_decomp (self, hdd_l):
        """
        months decomposition 
        hdd ... hdd list is in the form (a, a_1), (a,a_2), ... 
        same start for all 
        this is used to determine the volatility structure by months 
        """

        hdd_start = hdd_l[0][0] # start is the same for all lists 
        hdd_l_sorted = sorted(hdd_l, key=lambda hdd: hdd[1])  # sorted hdd_l
        hdd_overlap_l = {0: (hdd_start, hdd_l_sorted[0][1])} # first element
        last_ind = hdd_l_sorted[0][1]
        for e in enumerate(hdd_l_sorted[1:]):
            hdd_overlap_l[e[0]+1] = (last_ind, e[1][1]) # first index was already handled
            last_ind = e[1][1]
        return hdd_overlap_l


    def add_months(self, sourcedate, months):
        """
        adds months to sourcedate (in datetime format)
        """
        month = sourcedate.month - 1 + months
        year = sourcedate.year + month / 12
        month = month % 12 + 1
        day = min(sourcedate.day,calendar.monthrange(year,month)[1])
        return datetime.date(year,month,day)

    def months_index(self, HDD_date_l, sl_init ):
        """
        construct months index 
        HDD date_l is in the dictionary form {k: (date_s, nb_months)}
        sl_init ... initial value for (sigma, lambda)
        """

        months_input = [ (1, k+1) for x,k in HDD_date_l.values() ]
        sigma_lam_l = [ sl_init for m in months_input ]
        return {"month_decomp": months_input, 
                "sigma_lam": sigma_lam_l
                }
        

    def HD_calib_all(self, HDD_date_l, HDD_price_l, HDDO_price_l):
        """
        calibrates everything 
        """
        mi = self.months_index (HDD_date_l, self.sp) # mi ... month index 
        # mi_dec = month_decomp (mi["month_decomp"])
    
        sl_calib = [self.sp] * len(HDD_date_l)

        for nb_idx in np.arange(len(self.sp)): # go over all contracts 
            # do the calibration of sigma_lambda 
            print "calibrating ", nb_idx
            def opt_fct (sl):
                sl_calib[nb_idx] = sl
                nb_months = mi["month_decomp"][nb_idx][1]-1
                hdd1 = (self.HDD_real(nb_months, sl_calib, HDD_date_l) - HDD_price_l[nb_idx])**2
                hdd2 = (self.HDDO_real (HDDO_price_l[nb_idx][0], nb_months, sl_calib, HDD_date_l ) - HDDO_price_l[nb_idx][1])**2

                return hdd1 + hdd2

            p = openopt.NLP ( opt_fct, self.sp, lb=[0., - np.inf ] )
            sl = p.solve ('scipy_cobyla').xf
            sl_calib[nb_idx] = sl

        # self.sp = sl_calib

        return sl_calib

    def HDD_payoff(self, T_sm):
        return np.average (np.sum (np.maximum (65. - T_sm , 0.), axis = 0))

    # same as above, just computes a bunch of stuff on the device 
    def HDD_payoff_d (self, T_sm_d):
        T_sm_tmp_d = 65. - T_sm_d
        maximum_cuda ( T_sm_tmp_d ) # this changes T_sm_d, which souldnt
        # rowsum_cuda_notransfer(T_sm_tmp_d, rowsum_vec_d) # writes in rowsum_vec_d
        rowsum_vec_d = colsum_cuda_last(T_sm_tmp_d)

        # the reason why this is better is that it only carries over one number
        # implements return np.average( rowsum_vec_d.get() )    
        return average_reduction(rowsum_vec_d).get() / rowsum_vec_d.shape[0]


    def HDD_d(self, t_0, t_step, T_0, sp ):

        ttdp = (self.date_p - self.date_o).days / 365.25
        t_v = t_0 + t_step * np.arange(self.N_step) 
        T_m_v = self.T_m (ttdp + t_v )
        T_m_d1 = self.T_m_der (ttdp + t_v) # vector as well

        T_sm_d = self.T_sim_inn_d (T_m_v, T_m_d1, t_step, sp ) # T simulated matrix 
        return self.HDD_payoff_d( T_sm_d )

    def HDDO_payoff(self, K, T_sm, cp_ind = "c"):
        return np.average (np.maximum (np.sum (np.maximum (65. - T_sm , 0.), axis = 1) - K, 0.) )



    # HDD option real pricing 
    def HDDO_real (self, K, nb_months, sp_l, HDD_date_l, cp_ind = "c"):
    
        mi = self.months_index (HDD_date_l, sp_l[0]) # just need month_decom 
        mi_dec = self.month_decomp (mi["month_decomp"])
        sim_size = self.Z_m.shape[1]
        hdd_sim = np.zeros (( nb_months, sim_size))

        for n in np.arange (nb_months):
            t_month_start = self.add_months (self.date_s, n) 
            t_month_start_num = (t_month_start - self.date_p).days / 365.
            # T_0 = T_m_real (t_month_start, date_o, hp)
            t_step = 1. / 365.
            # N_step = (add_months(t_month_start, 1) - t_month_start).days 
            # COMPLETELY WRONG WRONG WRONG, PREVIOUS LINE IS CORRECT 
            # N_step = 31 
            sp = sp_l[self.month_into_sigma(n,mi_dec)] # mapping into sp_l[ n
            ttdp = (self.date_p - self.date_o).days / 365.25
            
            # following 3 lines used in both cpu/gpu comp.
            t_v = t_month_start_num + t_step * np.arange(self.N_step) 
            T_m_v = self.T_m (ttdp + t_v )
            T_m_d1 = self.T_m_der (ttdp + t_v ) # vector as well

            if self.gpu_ind == False:
                # Z_m = np.random.normal(size=(sim_size,N_step))
                T_sim = self.T_sim_inn (T_m_v, T_m_d1, t_step, sp ) 
                hdd_sim[n,:] = np.sum (np.maximum (65. - T_sim , 0.), axis = 0)

            else:
                T_sim_d = self.T_sim_inn_d ( T_m_v, T_m_d1, t_step, sp )
                T_sim_d = 65. - T_sim_d
                maximum_cuda(T_sim_d)
                rowsum_vec_d = colsum_cuda_last (T_sim_d)
                # write_vec_in_mat_col ( rowsum_vec_d, hdd_sim_d, n) # impl. : hdd_sim_d[n,:] = rowsum_vec_d
                hdd_sim[n,:] = rowsum_vec_d.get()

        return np.average (np.maximum ( np.sum (hdd_sim, axis =0 ) - K, 0.))


    def HDDO_histo (self, K, nb_months, HDD_date_l, cp_ind = 'c'):
        return self.HDDO_real (K, nb_months, [(0.,0.)] * len (HDD_date_l), HDD_date_l, 
                               cp_ind)




    def HDDO_payoff_d (self, K, T_sm_d, cp_ind = "c"):
        """
        same as above, just that most operations are performed on the device
        """
        T_sm_tmp_d = 65. - T_sm_d
        maximum_cuda(T_sm_tmp_d)
        colsum_cuda(T_sm_tmp_d)
        HDD_payoff_tmp = T_sm_tmp_d - K

        # implements: return np.average (np.maximum (HDD_payoff_tmp.get(), 0.) )
        return average_reduction (maximum_cuda (HDD_payoff_tmp)).get / HDD_payoff_tmp.shape[0]

    def T_par_inn (self, T_m_v, T_m_d, dt, sp):
        """
        fast simulation of the weather model partial innovations  
        """
        a = self.hp[5]
        sigma, sigma_lam = sp
        
        res = (T_m_d + a * T_m_v - sigma_lam ).reshape((len(T_m_d),1)) * dt + \
            sigma * np.sqrt (dt) * self.Z_m

        return res


    def T_par_inn_d (self, T_m_v, T_m_d1, dt, sp):
        """
        same as T_par_inn, except that it runs on GPU device 
        """

        a = self.hp[5]
        sigma, sigma_lam = sp

        # this implements ( *a _has_ be on the right )
        v = (T_m_d1 + T_m_v * a - sigma_lam ) * dt
    
        # Z_m_d NEEDS to have 0's in the first column
        # v = np.append ( T_0, ( T_m_d + a * T_m_v  - sigma_lam ) * dt )

        inn_d = self.Z_m_d * (sigma * np.sqrt (dt)) # mult. _has_ to be on the right (problems with pycuda)
        vtpm_cols ( v, inn_d, 'network_struct' ) # inn_d <- v_d + inn_d 

        return inn_d



    
    def T_sim_inn (self, T_m_v, T_m_d, t_step, sp):
        """
        simulation from partial innovations, FAST simulation 
        """
        # t_v = t_0 + t_step * np.arange(N_step) 
        T_s_1 = self.T_par_inn (T_m_v, T_m_d, t_step, sp)
    
        # THERE SHOULD BE T_0, perhaps we can circumvent this 
        # T_s_1 = np.append (T_0 * np.ones((Z_m.shape[0],1)), T_s_1, axis=1)
        a = self.hp[5]
        T_s_2 = T_s_1 * (1-a * t_step)**np.arange(self.N_step-1,-1,-1).reshape((self.N_step,1))

        return np.cumsum(T_s_2,axis=0) / (1-a*t_step)**np.arange(self.N_step-1,-1,-1).reshape((self.N_step,1))

    def T_sim_inn_d (self, T_m_v_v, T_m_d_v, t_step, sp ):
        # this below is inlined directly below 
        # T_s_1_d = T_par_inn_d ( T_m_v_d, T_m_d_d, t_step, sp, hp, Z_m_d, date_p, date_o)
        
        a = self.hp[5]
        sigma, sigma_lam = sp
        # this implements ( *a _has_ be on the right )
        v = (T_m_d_v + T_m_v_v * a - sigma_lam ) * t_step
        T_s_1_d = self.Z_m_d * (sigma * np.sqrt (t_step)) # mult. _has_ to be on the right (problems with pycuda)
        vtpm_cols ( v, T_s_1_d, 'network_struct' ) # inn_d <- v_d + inn_d 

        # appending from T_sim_inn is done in T_par_inn_d directly 
        # implements: T_s_1_d =  (1-a * t_step)**arange(N_step,-1,-1) * T_s_1_d 
        # TO IMPROVE BELOW:
        # N_step = 31 in this case, otherwise can be different
        # mult2 = gpa.to_gpu( (1-a * t_step)**(- arange(N_step, -1,-1)).astype(np.float32) )

        range_1 = ( 1. - a * t_step )**np.arange(self.N_step - 1, -1,-1)
        vtpm_cols ( range_1, T_s_1_d, 't' ) # multiply
        
        # following 2 statements implement the following 
        # np.cumsum(T_s_1_d.get(),axis=1)/ (1-a*t_step)**arange(N_step,-1,-1)
        colsum_cuda (T_s_1_d) # cumsum on cuda
        vtpm_cols ( 1./range_1, T_s_1_d, 't')
        
        return T_s_1_d

