# front office tolling model 
# see the front office doc. for other things 

# File defines:
import config 

import numpy
from numpy import *
import scipy
import scipy.optimize

import time
import openopt 
import pycuda.curandom 
import multiprocessing
import ctypes 


import tolling_fast
from pricers import cdf_vec 
import config.gpuarray as gpa 


# multi-threading version of tensor product 
tens_fast_mt_raw = ctypes.CDLL("/home/brumen/workspace/mrds/tp.so").tensor_prod_2
def tens_fast_mt(P_m, H_m, G_m, res_m):
    tens_fast_mt_raw ( ctypes.py_object(P_m),
                       ctypes.py_object(H_m),
                       ctypes.py_object(G_m),
                       ctypes.py_object(res_m))



# structure for tolling parameters 
class tolling_params():
    pass


# wrapper for the skew MRD model calibration function
def transition_wrap (arg, **kwarg):
    return tolling_model_MRD.transition_mtx_optim (*arg, **kwarg)


def step_wrap (arg, **kwarg):
    return tolling_model_MRD.step (*arg, **kwarg)

class tolling_model_lattice():
    """
    # backward induction algorithm on the lattice
    # params: parameters, which contain 
    #   .cuda ... indicator whether CUDA is present or not 
    #   .F0 ... forward price
    #   .K  ... cost of running a PP for that month 
    #   .sigma_F ... forward vol
    #   .sigma_C ... cash vol 
    #   .SC ... fixed startup costs 
    # debug_ind ... indicator whether debug is desired

    """

    def __init__ (self, params, debug_ind=False):
        #self.comp_ind = params.comp_ind # computational indicator: 0 ... slow CPU, 1 ... fast CPU, 2 ... MT CPU, 3 ... CUDA 
        self.cuda = params.cuda 
        self.nb_steps = params.nb_steps
        self.tolling_fast = params.tolling_fast # using the fast tensor routine from tolling_fast.pyx
        self.tolling_fast_mt = params.tolling_fast_mt # using raw multi-threading 
        self.params = params ## THIS CAN PERHAPS BE REMOVED LATER
        self.lattice_size = params.lattice_size
        self.minimum_downtime = params.MDT
        self.minimum_uptime = params.MUT
        self.maxCap = params.maxCap
        self.minCap = params.minCap
        self.debug_ind = debug_ind
        # initial values of power, fuel 
        self.F0 = params.F0 # forward price for that month 
        self.K = params.K  # cost of dispatch 
        self.sigma_F = params.sigma_F # forward vol for the month 
        self.sigma_C = params.sigma_C # cash vol for the month 
        # lattice construction 
        self.F_v = self.F0 * np.exp ( -1. + arange (double (params.lattice_size))/double (params.lattice_size) * 2. )
        self.F_vl = len (self.F_v)

        # BAD BAD BAD, HAS TO BE RESTRUCTURED
        self.running_profit_constr()
        self.zero_matrix = self.zero_pp()
        self.zero_matrix_d = gpa.to_gpu ( self.zero_pp() ).astype(float32)

        self.P_m = np.zeros ((self.F_vl, self.F_vl)) # transition matrix
        self.transition_mtx () # constructs a simple transition matrix 
        if self.cuda:
            self.P_m = gpa.to_gpu ( self.P_m ).astype(float32)

        # tmp. storage 
        if not self.cuda:
            self.work_curr_tmp = self.zero_pp() 
            self.idle_curr_tmp = self.zero_pp()
        else:
            self.work_curr_tmp = gpa.to_gpu ( self.zero_pp() ).astype(float32)
            self.idle_curr_tmp = gpa.to_gpu ( self.zero_pp() ).astype(float32)

        # the next power plant (pp) working condition 
        self.work_pp_next = []
        self.work_pp_curr = []
        self.idle_pp_next = []
        self.idle_pp_curr = []
        for Ut in range(self.minimum_uptime+1):
            if not self.cuda:
                self.work_pp_next.append ( self.zero_pp() )
                self.work_pp_curr.append ( self.zero_pp() )
            else:
                self.work_pp_next.append ( gpa.to_gpu (self.zero_pp() ).astype(float32) )
                self.work_pp_curr.append ( gpa.to_gpu (self.zero_pp() ).astype(float32) )

        for Dt in range(self.minimum_downtime+1):
            if not self.cuda:
                self.idle_pp_next.append ( self.zero_pp() )
                self.idle_pp_curr.append ( self.zero_pp() )
            else:
                self.idle_pp_next.append ( gpa.to_gpu ( self.zero_pp() ).astype (float32) )
                self.idle_pp_curr.append ( gpa.to_gpu ( self.zero_pp() ).astype (float32) )


    # construction of the zero matrix
    def zero_pp(self):
        return np.zeros ((self.F_vl, self.F_vl))


    def running_profit(self, po):
        """
        running profit from power plant 
        (this is a virtual function, can be overwritten)
        po ... profit object 
        """
        return (po.F - po.K) * po.A  # A ... capacity 

    def running_profit(self, F, market, params, delta_t=1. / 365.):
        """
        running profit from power plant
        (this is a virtual function, can be overwritten)
        """
        return (F - market.K) * params.maxCap * delta_t # A ... capacity




    def running_profit_constr (self):
        self.running_profit_mtx = self.zero_pp() 
        for F_1_ind in range(self.F_1_l):
            for F_2_ind in range(self.F_2_l):
                self.running_profit_mtx[F_1_ind,F_2_ind] = tolling_fast.running_profit_fast(self.F_1_v[F_1_ind], self.F_2_v[F_2_ind])
        self.running_profit_mtx_d = gpa.to_gpu (self.running_profit_mtx).astype (float32)

    # returns the running profit matrix 
    def running_profit_mtx (self):
        self.running_profit_mtx = self.zero_pp() 
        for F_1_ind in range(self.F_1_l):
            for F_2_ind in range(self.F_2_l):
                self.running_profit_mtx[F_1_ind,F_2_ind] = tolling_fast.running_profit_fast(self.F_1_v[F_1_ind], self.F_2_v[F_2_ind])
        return self.running_profit_mtx


    def transition_mtx_ln (self, F_v, delta_t):
        """
        transition between the stochastic states F_1, F_2 
        F_v ... lattice 
        transition from (F_v -> F_v ) 
        """

        F_va = np.diff (np.concatenate( (np.array([0.]), F_v) ) ) # augmented vector (va)
        F_va_l = len (F_va)

        self.P_m = cdf_vec ( (np.kron ( F_va, 1./ np.reshape (F_va, (F_va_l,1)) ) + 0.5 * self.sigma_C**2 * delta_t ) / ( self.sigma_C * np.sqrt (delta_t) ) )


    def transition_mtx_ln_newer(self, delta_t=1. / 365.):
        """
        transition between the stochastic states F_1, F_2
        transition from (F_v -> F_v )
        """

        F_va = np.diff(np.concatenate((np.array([0.]), self.F_v))) # augmented vector (va)
        F_va_l = len(F_va)

        # p'/p
        pdash_over_p = np.log(np.kron(self.F_v, 1. / np.reshape(self.F_v, (self.F_vl, 1))))

        cdf_transits = cdf_vec(
            (pdash_over_p + 0.5 * self.sigma_C ** 2 * delta_t ) / ( self.sigma_C * np.sqrt(delta_t) ))

        self.P_m = np.zeros((self.F_vl, self.F_vl + 1))
        self.P_m[:, 1:] = cdf_transits
        self.P_m = np.diff(self.P_m, axis=1)






    def transition_mtx_ln_blocks(self, p_dash, op, F_P, F_OP, sigma_P, sigma_OP, rho, t, delta_t=1./365.):
        """
        transition between blocks P(P(t+delta_t) < p' | OP(t) = op)
        """
        d1 = lambda p: (np.log(p_dash/p) + 0.5 * sigma_P**2 * delta_t) / (sigma_P * np.sqrt(delta_t) )
        z_p = lambda p: (np.log(p/F_P) + 0.5 * sigma_P**2 * t) / (sigma_P * np.sqrt(t) )
        z_op = lambda p: (np.log(op/F_OP) + 0.5 * sigma_OP**2 * t) / (sigma_OP * np.sqrt(t) )
        d2 = lambda p: (z_p(p) - rho * z_op(p)) / np.sqrt(1. - rho**2)

        f_int = lambda p: cdf_vec(d1(p)) * pdf_vec(d2(p)) / np.sqrt(1. - rho**2)

        return scipy.integrate.quad(f_int, 0., np.inf)[0] # value

    def transition_mtx_ln_blocks2(self, p_dash, op, F_P, F_OP, sigma_P, sigma_OP, rho, t, delta_t=1./365.):

        f_int = lambda p:  transition_mtx_ln_blocks_fast(p_dash, p, op, F_P, F_OP, sigma_P, sigma_OP,
                                                         rho, t, delta_t)

        return scipy.integrate.quad(f_int, 0., np.inf)[0]


    def transition_mtx_ln_blocks_sg(self, p_dash, op, F_P, F_OP, sigma_P, sigma_OP, rho, t, delta_t=1./365.,
                                    sg_level = 15):
        """
        sparse grid integration
          level ... sg level, how fine the sparse grid gets
        """

        f_int = lambda v:  transition_mtx_ln_blocks_fast_internal (p_dash, v, op, F_P, F_OP, sigma_P, sigma_OP,
                                                                   rho, t, delta_t)

        return sg.sg_quad(1, sg_level, f_int)



    def transition_mtx_ln_blocks_sg_fast_helper_old(self, v, p_dash, op, F_P, F_OP, sigma_P, sigma_OP, rho, t, delta_t=1./365.,
                                                sg_level = 15):

        z_op = (np.log(op/F_OP) + 0.5 * sigma_OP**2 * t) / (sigma_OP * np.sqrt(t) )
        # p is vector
        # z_p = (np.log(p/F_P) + 0.5 * sigma_P**2 * t) / (sigma_P * np.sqrt(t) )
        # v = (z_p - rho * z_op) / sqrt(1. - rho**2)
        d1 = (np.log(p_dash/F_P) + 0.5 * sigma_P**2 * (t+delta_t) - np.sqrt(1. - rho**2) * sigma_P * np.sqrt(t) * v
              - rho * z_op * sigma_P * np.sqrt(t)
             ) / (sigma_P * np.sqrt(delta_t) )

        return cdf_vec(d1) # n(v) is omitted, as Guass-Hermite handles it



    def transition_mtx_ln_blocks_all_old(self, step_nb):
        """
        constructs a transition matrix
        """
        if step_nb == len (self.hours_seq) -1: # last step, no transition
            return np.zeros ((self.lattice_size, self.lattice_size))
        else:
            F_next_v = self.lattice_seq[step_nb] # next lattice
            F_curr_v = self.lattice_seq[step_nb+1] # curr. lattice
            P_m = np.zeros ((self.lattice_size, self.lattice_size))
            P_m_tmp = np.zeros (self.lattice_size + 1) # tmp. mtx for taking differences
            for (F_curr_idx, F_curr) in enumerate(F_curr_v):
                for (F_next_idx, F_next) in enumerate(F_next_v):
                    P_m[F_curr_idx, F_next_idx] = self.transition_mtx_ln_blocks_sg_fast(F_next, F_curr,
                                                                                self.market_seq[step_nb+1]["fwd"],
                                                                                self.market_seq[step_nb]["fwd"],
                                                                                self.market_seq[step_nb+1]["sigma_C"],
                                                                                self.market_seq[step_nb]["sigma_C"],
                                                                                0.9, # rho WRONG WRONG WRONG
                                                                                np.sum(self.t_diff_seq[:step_nb+2]),
                                                                                self.t_diff_seq[step_nb+2] # WRONG WRONG WRONG
                                                                                )
                P_m_tmp[1:] = P_m[F_curr_idx,:]
                P_m[F_curr_idx,:] = np.diff(P_m_tmp)

            #if self.debug_ind:
            print "Finished generating matrix = ", np.sum(P_m, axis = 1)

            return P_m
        
    # multi-threading version of the transition matrix 
    def transition_mtx_multi (self):
        nb_cores = multiprocessing.cpu_count() 
        print "Using ", nb_cores , " cores."
        pool = multiprocessing.Pool(processes=nb_cores) 
        print "Starting worker threads"
        C = pool.map ( transition_wrap, zip ([self] * len(self.F_1_v), 
                                             xrange(len(self.F_1_v)) ) )

        return C 




    # dynamic step 
    def step(self, Ut_curr, Dt_curr):

        def tensor_slow (P_m, H_m, G_m):
            # P_m, H_m, G_m, res_m have to be of appropriate sizes 

            res_m = self.zero_pp()
            # if this is OK, the rest goes through
            s = P_m.shape[0]
            for F_1_ind in range(s):
                for F_2_ind in range(s):
                    res_m[F_1_ind,F_2_ind] = sum (P_m[F_1_ind,F_2_ind,:,:] * H_m) + G_m[F_1_ind, F_2_ind] 
            # everything is attributed to res_m 
            return res_m


        # beginning perid 
        if ( (Ut_curr == 0) and (Dt_curr == 0) ):
            self.work_pp_curr[0] = tensor_slow ( self.P_m, self.work_pp_next[1], self.running_profit_mtx() )
            self.idle_pp_curr[0] = tensor_slow ( self.P_m, self.idle_pp_next[1], self.zero_pp() )

        if (Ut_curr > 0): # working plant, continue operations
            if (Ut_curr == self.minimum_uptime):
                self.work_curr_tmp = tensor_slow (self.P_m, self.work_pp_next[self.minimum_uptime], self.running_profit_mtx() )
            else:
                self.work_curr_tmp = tensor_slow (self.P_m, self.work_pp_next[Ut_curr+1], self.running_profit_mtx() )
                        
            self.idle_curr_tmp = tensor_slow (self.P_m, self.idle_pp_next[0], self.running_profit_mtx() )

            self.work_pp_curr[Ut_curr] = maximum (self.idle_curr_tmp, self.work_curr_tmp)


        else: # idle plant, Dt_curr
            if (Dt_curr == self.minimum_downtime):
                self.idle_curr_tmp = tensor_slow (self.P_m, self.idle_pp_next[self.minimum_downtime], self.zero_pp() )
            else:
                self.idle_curr_tmp = tensor_slow (self.P_m, self.idle_pp_next[Dt_curr+1], self.zero_pp())
            # restarting the plant 
            self.work_curr_tmp = tensor_slow (self.P_m, self.work_pp_next[0], self.zero_pp())

            self.idle_pp_curr[Dt_curr] = maximum (self.idle_curr_tmp, self.work_curr_tmp) 

    # fast version of the routines above 
    def step_fast(self, Ut_curr, Dt_curr):

        # beginning perid 
        if ( (Ut_curr == 0) and (Dt_curr == 0) ):
            tolling_fast.tensor_fast (self.P_m, self.work_pp_next[1],self.running_profit_mtx, self.work_pp_curr[0])
            tolling_fast.tensor_fast (self.P_m, self.idle_pp_next[1], self.zero_matrix, self.idle_pp_curr[0])            

        if (Ut_curr > 0): # working plant, continue operations
            if (Ut_curr == self.minimum_uptime):
                tolling_fast.tensor_fast (self.P_m, self.work_pp_next[self.minimum_uptime], self.running_profit_mtx, self.work_curr_tmp)
                
            else:
                tolling_fast.tensor_fast (self.P_m, self.work_pp_next[Ut_curr+1], self.running_profit_mtx, self.work_curr_tmp)

            tolling_fast.tensor_fast (self.P_m, self.idle_pp_next[0], self.running_profit_mtx, self.idle_curr_tmp) 

            self.work_pp_curr[Ut_curr] = maximum (self.idle_curr_tmp, self.work_curr_tmp) 


        else: # idle plant, Dt_curr
            if (Dt_curr == self.minimum_downtime):
                tolling_fast.tensor_fast (self.P_m, self.idle_pp_next[self.minimum_downtime], self.zero_matrix, self.idle_curr_tmp)
            else:
                tolling_fast.tensor_fast (self.P_m, self.idle_pp_next[Dt_curr+1], self.zero_matrix, self.idle_curr_tmp)
                        
            # restarting the plant 
            tolling_fast.tensor_fast (self.P_m, self.work_pp_next[0], self.zero_matrix, self.work_curr_tmp)

            self.idle_pp_curr[Dt_curr] = maximum (self.idle_curr_tmp, self.work_curr_tmp) 

    # fast version of the routines above using multi-threading 
    def step_fast_mt(self, Ut_curr, Dt_curr):

        # beginning perid 
        if ( (Ut_curr == 0) and (Dt_curr == 0) ):
            tens_fast_mt (self.P_m, self.work_pp_next[1],self.running_profit_mtx, self.work_pp_curr[0])
            tens_fast_mt (self.P_m, self.idle_pp_next[1], self.zero_matrix, self.idle_pp_curr[0])            

        if (Ut_curr > 0): # working plant, continue operations
            if (Ut_curr == self.minimum_uptime):
                tens_fast_mt (self.P_m, self.work_pp_next[self.minimum_uptime], self.running_profit_mtx, self.work_curr_tmp)
                
            else:
                tens_fast_mt (self.P_m, self.work_pp_next[Ut_curr+1], self.running_profit_mtx, self.work_curr_tmp)

            tens_fast_mt (self.P_m, self.idle_pp_next[0], self.running_profit_mtx, self.idle_curr_tmp) 

            self.work_pp_curr[Ut_curr] = maximum (self.idle_curr_tmp, self.work_curr_tmp) 


        else: # idle plant, Dt_curr
            if (Dt_curr == self.minimum_downtime):
                tens_fast_mt (self.P_m, self.idle_pp_next[self.minimum_downtime], self.zero_matrix, self.idle_curr_tmp)
            else:
                tens_fast_mt (self.P_m, self.idle_pp_next[Dt_curr+1], self.zero_matrix, self.idle_curr_tmp)
                        
            # restarting the plant 
            tens_fast_mt (self.P_m, self.work_pp_next[0], self.zero_matrix, self.work_curr_tmp)

            self.idle_pp_curr[Dt_curr] = maximum (self.idle_curr_tmp, self.work_curr_tmp) 


    # dynamic step using cuda 
    def step_cuda(self, Ut_curr, Dt_curr):
        tensor_string = open ("tensor.cu").read()
        #tensor_prod_mod = config.SourceModule(tensor_string%{"lattice_size": self.lattice_size})
        tensor_prod_mod = config.SourceModule(tensor_string)
        tensor_chosen = "tensor_P_m_alt" # can also be tensor_P_m
        tensor_prod = tensor_prod_mod.get_function (tensor_chosen) # extracting compute vol function

        # selection of the block size and grid size 
        if tensor_chosen == "tensor_P_m":
            block_sel = (self.lattice_size,1,1)
            grid_sel = (self.lattice_size,1)
        else: # testing the different cuda kernels 
            block_sel = (32,1,1) # 32 threads - warp 
            grid_sel = (self.lattice_size,self.lattice_size)

        # beginning period 
        if ( (Ut_curr == 0) and (Dt_curr == 0) ):
            tensor_prod ( self.P_m, self.work_pp_next[1], self.running_profit_mtx_d, self.work_pp_curr[0],  
                          block=block_sel, grid=grid_sel )
            tensor_prod ( self.P_m, self.idle_pp_next[1], self.zero_matrix_d, self.idle_pp_curr[0],
                          block=block_sel, grid=grid_sel )

        elif (Ut_curr > 0): # working plant, continue operations
            if (Ut_curr == self.minimum_uptime):
                tensor_prod ( self.P_m, self.work_pp_next[self.minimum_uptime], self.running_profit_mtx_d, self.work_curr_tmp, 
                              block=block_sel, grid=grid_sel ) 

            else:
                tensor_prod ( self.P_m, self.work_pp_next[Ut_curr+1], self.running_profit_mtx_d, self.work_curr_tmp, 
                              block=block_sel, grid=grid_sel ) 

            tensor_prod ( self.P_m, self.idle_pp_next[0], self.running_profit_mtx_d, self.idle_curr_tmp, 
                          block=block_sel, grid=grid_sel ) 
                        
            self.work_pp_curr[Ut_curr] = gpa.maximum (self.idle_curr_tmp, self.work_curr_tmp) 

        else: # idle plant, Dt_curr
            if (Dt_curr == self.minimum_downtime):
                tensor_prod ( self.P_m, self.idle_pp_next[self.minimum_downtime], self.zero_matrix_d, self.idle_curr_tmp, 
                              block=block_sel, grid=grid_sel )
            else:
                tensor_prod ( self.P_m, self.idle_pp_next[Dt_curr+1], self.zero_matrix_d, self.idle_curr_tmp, 
                              block=block_sel, grid=grid_sel ) 

            # restarting the plant 
            tensor_prod ( self.P_m, self.work_pp_next[0], self.zero_matrix_d, self.work_curr_tmp, 
                          block=block_sel, grid=grid_sel ) 
            
            self.idle_pp_curr[Dt_curr] = gpa.maximum (self.idle_curr_tmp, self.work_curr_tmp) 

    def terminal_pp_value(self):
        for work_pp_ind in range(self.minimum_uptime):
            if not self.cuda:
                self.work_pp_next[work_pp_ind] = self.zero_pp ()
            else:
                self.work_pp_next[work_pp_ind] = gpa.to_gpu ( self.zero_pp () ).astype(float32)
        for idle_pp_ind in range(self.minimum_downtime):
            if not self.cuda: 
                self.idle_pp_next[idle_pp_ind] = self.zero_pp ()
            else:
                self.idle_pp_next[idle_pp_ind] = gpa.to_gpu ( self.zero_pp () ).astype (float32)

    # all steps for 1 time step 
    def all_one_steps(self):
        if not self.cuda:
            if self.tolling_fast: # fast version of tensor product
                self.step_fast(0,0)
            elif self.tolling_fast_mt :
                self.step_fast_mt (0,0)
            else:
                self.step(0,0)
        else:
            self.step_cuda(0,0)
                
        for Ut in range (1,self.minimum_uptime+1):
            if not self.cuda:
                if self.tolling_fast:
                    self.step_fast(Ut,0)
                elif self.tolling_fast_mt:
                    self.step_fast_mt (Ut, 0)
                else:
                    self.step(Ut,0)
            else:
                self.step_cuda(Ut,0)
        for Dt in range (1,self.minimum_downtime+1):
            if not self.cuda:
                if self.tolling_fast:
                    self.step_fast(0,Dt)
                elif self.tolling_fast_mt:
                    self.step_fast_mt (0, Dt)
                else:
                    self.step(0,Dt)
            else:
                self.step_cuda(0,Dt)

    # multithreading version of the above function
    def all_one_steps_mt(self):
        nb_cores = multiprocessing.cpu_count() 
        print "Using ", nb_cores , " cores."
        pool = multiprocessing.Pool(processes=nb_cores) 
        print "Starting multithread one step"
        pool.map ( step_wrap, \
                       zip ([self] * (self.minimum_uptime+1), range(self.minimum_uptime+1), \
                                [0] * (self.minimum_uptime+1) ) )
        pool.map ( step_wrap, \
                       zip ([self] * (self.minimum_downtime+1), \
                                [0] * (self.minimum_downtime+1), range(self.minimum_downtime+1) ) )
        




    def overwrite_next_w_curr(self):
        for work_pp_ind in range(self.minimum_uptime):
            self.work_pp_next[work_pp_ind] = self.work_pp_curr [work_pp_ind]
        for idle_pp_ind in range(self.minimum_downtime):
            self.idle_pp_next[idle_pp_ind] = self.idle_pp_curr [work_pp_ind]

            
    def multiple_steps(self, n):
        print "Starting steps"
        for ind in range(n):
            t1 = time.time()
            self.all_one_steps()
            self.overwrite_next_w_curr()
            t2 = time.time()
            print "Step ", ind, " finished in ", (t2-t1)*1000.0

    # multi-threading version of the above
    def multiple_steps_mt(self, n):
        for ind in range(n):
            self.all_one_steps_mt()
            self.overwrite_next_w_curr()

    # select the value from the power plant 
    def tolling_value (self):
        self.multiple_steps(self.nb_steps)
        if not self.cuda:
            return self.work_pp_curr[0][self.lattice_size/2, self.lattice_size/2]
        else: 
            curr_mtx = self.work_pp_curr[0].get()
            return curr_mtx[self.lattice_size/2, self.lattice_size/2]
