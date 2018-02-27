#
# File defines:
#   mrd skew model for commodities (state reference)
#   a general diffusion model

import config 

from numpy import *
import numpy as np
import numpy.random
import scipy
from scipy.optimize import brenth, fmin
import openopt # optimization solver
import pycuda.driver as drv
import pycuda.curandom
import pycuda.gpuarray as gpa
import pycuda.cumath
import pycuda.reduction
#from cudanormal import cudanormal
import datetime
import calendar
#import timeit
import time
import cublas

import timing

# vector + matrix slicing kernel - vpm
# vector * matrix slicing kernel - vtm
# TO CORRECT: N_STEP IS FIXED. 

vtpm_code = """
__global__ void vpm (float *v, float *m, int nb_cols, int nb_rows, int to_do_rows ) {
  int ind1, res_idx;
  int th_idx = threadIdx.x;
  int th_bl_idx = th_idx + blockIdx.x * blockDim.x;
  __shared__ float v_cache[31];
  v_cache[th_idx] = v[th_idx];

  for (ind1 = 0; ind1 < (to_do_rows); ind1 = ind1 + 1) {
    res_idx = ind1 * (nb_cols) * 65535 + th_bl_idx;
    if ( res_idx < ( nb_rows ) * (nb_cols) )
      m[res_idx] += v_cache[th_idx];
  }
}

__global__ void vtm (float *v, float *m, int nb_cols, int nb_rows, int to_do_rows ) {
  int ind1, res_idx;
  int th_idx = threadIdx.x;
  int th_bl_idx = th_idx + blockIdx.x * blockDim.x;
  __shared__ float v_cache[31];
  v_cache[th_idx] = v[th_idx];

  for (ind1 = 0; ind1 < (to_do_rows); ind1 += 1) {
    res_idx = ind1 * (nb_cols) * 65535 + th_bl_idx;
    if ( res_idx < (nb_rows) * (nb_cols) )
      m[res_idx] *= v_cache[th_idx];
  }
}
"""

# experimental code
vtpm_module = config.SourceModule(vtpm_code)
vpm_f = vtpm_module.get_function ("vpm") # vector + matrix function 
vtm_f = vtpm_module.get_function ("vtm") # vector + matrix function 



def vtpm(v,m, tm_ind = 'p'):
    """ 
    vector times matrix, by rows 
    tm_ind = p ... for summation (plus)
    tm_ind = t ... for multiplication (times)
    """
    m_cols = m.shape[1]
    m_rows = m.shape[0]
    nb_launches = m_rows / 65535 +1 # this is an integer

    block_dims = (m_cols,1,1)
    vtpm_f = {'t': vtm_f, 'p': vpm_f} # for a single launch this works best

    if m_rows / 65535 > 0: 
        grid_dims = (65535, 1)
        vtpm_f[tm_ind](v,m, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
                       block=block_dims, grid= grid_dims)
    else:
        grid_dims = (m_rows, 1)
        vtpm_f[tm_ind]( v, m, np.int32(m_cols), np.int32(m_rows), np.int32(1),
                         block = block_dims, grid = (grid_dims) )
        
        

#vtm_module = config.SourceModule(vtm_code)
#vtm_f = vtm_module.get_function ("vtm") # vector + matrix function 
#vtm = lambda v,m: vtm_f(v,m, block=(m.shape[1],1,1), grid=(m.shape[0],1))

# computes a**v, where a is a number, v is a vector 
# --- became part of gpuarray - apow function 
#vpow_module = config.SourceModule(vpow_code)
#vpow_f = vpow_module.get_function ("vpow") # vector + matrix function 
#vpow = lambda a,v: vpow_f(a,v, block=(1,1,1), grid=(len(v),1))


cumsum_cuda_code = """
__global__ void cumsum_cuda (float *M, int nb_cols, int nb_rows, int to_do_rows) {

  int ind1, ind2, curr_row;
  float curr_val;
  int row_start_idx;

  for (ind1 = 0; ind1 < (to_do_rows); ind1 += 1) { /* traversing rows */
    row_start_idx = ind1 * nb_cols * 65535 + blockIdx.x * nb_cols;
    curr_row = ind1 * 65535 + blockIdx.x;
    if ( curr_row < nb_rows ) {
      curr_val =  M[row_start_idx]; 
      for (ind2 = 1; ind2 < nb_cols; ind2 += 1) { /* traversing individual row across cols */
        curr_val = curr_val + M[row_start_idx + ind2];
        M[row_start_idx + ind2] = curr_val;
      }
    }
  }
}
"""
cs_module = config.SourceModule(cumsum_cuda_code)
cs_cuda_f = cs_module.get_function ("cumsum_cuda") 

# row cumsum function (on cuda, m is a matrix), replaces m_d with 
def cumsum_cuda(m_d):

    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1 # this is an integer

    block_dims = (1,1,1)
    grid_dims = (65535, 1) # rows
    cs_cuda_f(m_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
              block=block_dims, grid= grid_dims)

    return 0


# maximum code
# maximum function, does max (M,0) by elements 
maximum_cuda_code = """
__global__ void maximum_cuda ( float *M, int nb_cols, int nb_rows, int to_do_rows ) {

  int ind1;
  int res_idx;

  for (ind1 = 0; ind1 < to_do_rows; ind1 = ind1 + 1) {
    res_idx = ind1 * nb_cols * 65535 + threadIdx.x + blockIdx.x * blockDim.x;
    
    if ( res_idx < nb_rows * nb_cols )
      M[res_idx] = max (M[res_idx], 0.0);
  }
}
"""


# maximum_cuda, computes max (M,0), overwrites M
maximum_cuda_module = config.SourceModule(maximum_cuda_code)
maximum_cuda_f = maximum_cuda_module.get_function ("maximum_cuda") 

def maximum_cuda (m_d):

    if len(m_d.shape) == 1: # vector, not matrix 
        m_cols = 1
    else:
        m_cols = m_d.shape[1]
    
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1 # this is an integer
    block_dims = (m_cols,1,1) # cols
    grid_dims = (65535, 1) # rows

    maximum_cuda_f(m_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
                   block=block_dims, grid= grid_dims)



# write the cumsum of M, the last column in v 
# rows is large, cols is much smaller 
rowsum_cuda_code_backup = """
__global__ void rowsum_cuda (float *M, float *v, 
                             int nb_cols, int nb_rows, int to_do_rows) {

  int ind1, ind2, curr_row;
  float curr_val;
  int row_start_idx;
  // __shared__ float curr_row[31];
  
  for (ind1 = 0; ind1 < to_do_rows; ind1 += 1) { /* traversing rows */
    row_start_idx = ind1 * nb_cols * 65535 + blockIdx.x * nb_cols;
    curr_row = ind1 * 65535 + blockIdx.x;
    if ( curr_row < nb_rows ) {
      curr_val =  M[row_start_idx]; 
      for (ind2 = 1; ind2 < nb_cols; ind2 += 1) { /* traversing individual row across cols */
        curr_val = curr_val + M[row_start_idx + ind2];

      }
    v[curr_row] = curr_val;
    }
  }
}
"""




rowsum_cuda_code = """
__global__ void rowsum_cuda (float *M, float *v, 
                             int nb_cols, int nb_rows, int to_do_rows) {

  int ind1, ind2;
  int row_start_idx;
  int curr_row = blockIdx.x * blockDim.x + threadIdx.x;

  // __shared__ float curr_col[];

  /* initial step */ 
  if (curr_row < nb_rows )
    v[curr_row] = M[curr_row]; 

  /* for each column */
  for (ind2 = 1; ind2 < nb_cols; ind2 += 1)
    if ( curr_row < nb_rows ) 
      v[curr_row] +=  M[ind2 * nb_rows + curr_row]; 

  /* final assignement */
  //if (curr_row < nb_rows )
  //  v[curr_row] = 1.0; // curr_col[curr_row];

}
"""


rs_module = config.SourceModule(rowsum_cuda_code)
rs_cuda_f = rs_module.get_function ("rowsum_cuda") 

# row sum cuda - same as cumsum cuda, just that it returns last col of the matrix, 
# a final cummulative sum 
def rowsum_cuda (m_d, ones_d, rs_res_d):

    # m_cols = m_d.shape[1]
    # m_rows = m_d.shape[0]
    # ones_d = gpa.to_gpu ( np.ones ((m_cols,1)).astype(np.float32) )
    cublas.cublasSgemv_d (1., m_d, ones_d, 0., rs_res_d)


    #return rs_res_d


def rowsum_cuda_backup(m_d):

    rs_res_d = gpa.zeros ( m_d.shape[0], np.float32) # row-sum result on device 
    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1 # this is an integer

    block_dims = (1,1,1)
    grid_dims = (65535, 1) # rows
    rs_cuda_f(m_d, rs_res_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
              block=block_dims, grid= grid_dims)
    
    return rs_res_d



def rowsum_cuda_notransfer(m_d, rs_res_d):
    """ 
    sums of rows of the matrix m_d are written in v_d, which is already a 
    device vector 
    """
    # size rs_res_d == m_rows

    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1  # this is an integer

    block_dims = (1,1,1)
    grid_dims = (65535, 1) # rows
    rs_cuda_f(m_d, rs_res_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
              block=block_dims, grid= grid_dims)
    
    return rs_res_d


# BACKUP COPY, DO NOT TOUCH
def rowsum_cuda_notransfer_backup(m_d, rs_res_d):
    """ 
    sums of rows of the matrix m_d are written in v_d, which is already a 
    device vector 
    """
    # size rs_res_d == m_rows

    m_cols = m_d.shape[1]
    m_rows = m_d.shape[0]
    nb_launches = m_rows / 65535 +1  # this is an integer

    block_dims = (1,1,1)
    grid_dims = (65535, 1) # rows
    rs_cuda_f(m_d, rs_res_d, np.int32(m_cols), np.int32(m_rows), np.int32(nb_launches), 
              block=block_dims, grid= grid_dims)
    
    return rs_res_d






# THIS IS WRONG BAD PRACTICE
sim_size = 400000
N_step = 31
ftol = 3.
# Z_m, Z_m_d generated only once, as they are no longer needed later 
#Z_m = np.random.normal(size=(sim_size,31))
#Z_m_d = gpa.to_gpu ( np.append ( np.zeros((sim_size, 1)), np.random.normal(size=(sim_size,31)), axis=1).astype(np.float32) )
#Z_m_d = gpa.to_gpu ( np.append ( np.zeros((sim_size, 1)), Z_m, axis=1).astype(np.float32) )
#Z_m_d = gpa.to_gpu (Z_m.astype(np.float32))

# THIS BELOW IS BAD
#range_gpu = gpa.to_gpu( arange(N_step-1, -1,-1).astype(np.float32) )
#range_gpum1 = gpa.to_gpu( - arange(N_step-1, -1,-1).astype(np.float32) )
# THIS IS WRONG BELOW, SHOULD BE N_step, but for now, we leave it
#range_gpu_inv = gpa.to_gpu (arange(N_step).astype(np.float32))  # inverse, used in T_par_inn_d
rowsum_vec_d = gpa.zeros ( sim_size, np.float32) # row-sum result on device 
#hdd_sim_sum_d = gpa.zeros ( sim_size, np.float32) # hdd_simulation sum 
T_help_d = gpa.zeros ( N_step, np.float32)
T_d_help_d = gpa.zeros (N_step, np.float32)
#ones_d = gpa.to_gpu ( np.ones ((N_step,1)).astype(np.float32) )
#rs_res_d = gpa.zeros ( sim_size, np.float32) # row-sum result on device 

# average function
def T_m (t, hp):
    A, B, C, omega, phi, a = hp # a is the mean reversion params
    #print "c_mine = ", sin(omega * t + phi)
    return A + B * t + C * sin(omega * t + phi)


sin_cos_fast_code = """
__global__ void sin_fast(float *x, float *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = sinf (x[idx]);
}

__global__ void cos_fast(float *x, float *y, int x_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < x_size) 
      y[idx] = cosf (x[idx]);
}
"""

sin_cos_fast_module = config.SourceModule(sin_cos_fast_code)
sin_cos_f = {'sin': sin_cos_fast_module.get_function ("sin_fast"),
             'cos': sin_cos_fast_module.get_function ("cos_fast")}

def sin_cos_d ( x, y, sin_cos = 'sin'):
    """
    implements the sin, cos on x, and writes it in y LLL
    """
    x_len = len (x)
    sin_cos_f[sin_cos](x, y, np.int32(x_len), block=(512,1,1), grid=(x_len/512 + 1,1) )


# average function for t_d on the device
def T_m_d (t_d, hp):
    A, B, C, omega, phi, a = hp # a is the mean reversion params
    #return A + B * t_d + C * pycuda.cumath.sin(omega * t_d + phi)
    t_d1 = t_d *omega + phi
    sin_cos_d(t_d1, T_help_d)
    #print "entering T_m_d"
    #print "t_d = ", t_d
    #print "d_real = ", t_d.get() * omega + phi
    #print "d_mine = ", t_d1
    return A + B * t_d + C * T_help_d


def T_m_real (t_date, t_origin, hp):
    t_num = (t_date - t_origin).days / 365.25
    return T_m (t_num, hp)

def T_m_der (t, hp):
    A, B, C, omega, phi, a = hp
    return B + C * cos (omega * t + phi) * omega

# same as above function, it just does the t_d on the device
def T_m_der_d (t_d, hp):
    A, B, C, omega, phi, a = hp
    # implements: return B + C * pycuda.cumath.cos (omega * t_d + phi) * omega
    sin_cos_d(omega * t_d + phi, T_d_help_d, 'cos')
    return B + ( C * omega ) * T_d_help_d


def T_m_der_real (t_date, t_origin, hp):
    t_num = (t_date - t_origin).days / 365.25
    return T_m_der (t_num, hp)


# step of simulation 
# sp ... simulation parameters (sigma, lam) 
# hp ... historical parameters, see T_m
# Z is a vector of same len. as T_t_v
def T_step (t, dt, T_t_v, sp,  hp, Z):
    T_m_v = T_m (t, hp)
    T_m_d = T_m_der (t,hp)
    a = hp[5]
    sigma, lam = sp
    return T_t_v + (T_m_d + a * (T_m_v - T_t_v ) - lam * sigma ) * dt + sigma * sqrt (dt) * Z

# complete simulates the whole process, slow 
def T_sim (t_0, t_step, N_step, T_0, sp, hp, Z_m):
    T_0_v = T_0 * np.ones(Z_m.shape[0])
    T_s = np.zeros ((Z_m.shape[0], Z_m.shape[1]+1)) # simulated matrix (
    T_s[:,0] = T_0_v
    for n in arange(1,N_step+1):
        T_s[:,n] = T_step(t_step * n, t_step, T_s[:,n-1], sp, hp, Z_m[:,n-1])
    
    return T_s

def HDD(t_0, t_step, N_step, T_0, sp, hp, Z_m, date_p, date_o):
    ttdp = (date_p - date_o).days / 365.25
    t_v = t_0 + t_step * np.arange(N_step) 
    T_m_v = T_m (ttdp + t_v, hp)
    T_m_d1 = T_m_der (ttdp + t_v,hp) # vector as well

    T_sm = T_sim_inn ( T_m_v, T_m_d1, t_step, N_step, sp, hp, Z_m, date_p, date_o) # T simulated matrix 
    return HDD_payoff(T_sm)

# maps month n = 0, 1, 2, 3 into sp_l index when mi_dec is given 
def month_into_sigma(n, mi_dec):
    np1 = n+1
    return sum ([ m[1]<=np1 for m in mi_dec.values() ])

# incorporates correct handling of dates 
# sp_l, hp_l are lists of simulation parameters, historical parameters for months 
# date_p ... pricing date (datetime format)
# date_s ... start date, 
# date_o ... origin date (check what that really is)
# nb. months ... number of months of the HDD
def HDD_real (date_p, date_s, nb_months, sp_l, hp, HDD_date_l, date_o, Z_m, Z_m_d, range_gpu, range_gpum1, range_gpu_inv, gpu_ind = False):

    mi = months_index (HDD_date_l, sp_l[0]) # just need month_decom 
    mi_dec = month_decomp (mi["month_decomp"])

    hdd_val = 0.
    for n in arange (nb_months):
        t_month_start = add_months (date_s, n) 
        t_month_start_num = (t_month_start - date_p).days / 365.
        
        T_0 = T_m_real (t_month_start, date_o, hp)
        t_step = 1. / 365.
        N_step = (add_months(t_month_start, 1) - t_month_start).days 
        sp = sp_l[month_into_sigma(n,mi_dec)] # mapping into sp_l[ n
        if gpu_ind == False:
            hdd_val += HDD( t_month_start_num, t_step, N_step, T_0, sp, hp, Z_m[:,:N_step], date_p, date_o)
        else:
            # 31 IS WRONG, due to impossible slicing in gpuarray
            hdd_val += HDD_d( t_month_start_num, t_step, 31, T_0, sp, hp, Z_m_d, 
                              range_gpu, range_gpum1, range_gpu_inv, date_p, date_o)

    return hdd_val

def HDD_histo (date_p, date_s, nb_months, hp, HDD_date_l, date_o, 
               Z_m, Z_m_d, range_gpu, range_gpum1, range_gpu_inv, gpu_ind = False):

    sp_l = [(0.,0.)] * len(HDD_date_l)
    return HDD_real (date_p, date_s, nb_months, sp_l, hp, HDD_date_l, date_o, 
                     Z_m, Z_m_d, range_gpu, range_gpum1, range_gpu_inv, 
                     gpu_ind)


# months decomposition 
# hdd ... hdd list is in the form (a, a_1), (a,a_2), ... 
# same start for all 
# this is used to determine the volatility structure by months 
def month_decomp (hdd_l):
    hdd_start = hdd_l[0][0] # start is the same for all lists 
    hdd_l_sorted = sorted(hdd_l, key=lambda hdd: hdd[1])  # sorted hdd_l
    hdd_overlap_l = {0: (hdd_start, hdd_l_sorted[0][1])} # first element
    last_ind = hdd_l_sorted[0][1]
    for e in enumerate(hdd_l_sorted[1:]):
        hdd_overlap_l[e[0]+1] = (last_ind, e[1][1]) # first index was already handled
        last_ind = e[1][1]
    return hdd_overlap_l

# adds months to sourcedate (in datetime format)
def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month / 12
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

# construct months index 
# HDD date_l is in the dictionary form {k: (date_s, nb_months)}
# sl_init ... initial value for (sigma, lambda)
def months_index( HDD_date_l, sl_init ):
    months_input = [ (1, k+1) for x,k in HDD_date_l.values() ]
    sigma_lam_l = [ sl_init for m in months_input ]
    return {"month_decomp": months_input, 
            "sigma_lam": sigma_lam_l
            }
        
# calibrates everything 
def HD_calib_all(date_p, date_s, HDD_date_l, HDD_price_l, HDDO_price_l, hp, sl_init, date_o, 
                 Z_m, Z_m_d, range_gpu, range_gpum1, range_gpu_inv, gpu_ind = False):

    mi = months_index (HDD_date_l, sl_init) # mi ... month index 
    #mi_dec = month_decomp (mi["month_decomp"])
    
    sl_calib = [sl_init] * len(HDD_date_l)

    for nb_idx in arange(len(sl_init)): # go over all contracts 
        # do the calibration of sigma_lambda 
        print "calibrating ", nb_idx
        def opt_fct (sl):
            sl_calib[nb_idx] = sl
            nb_months = mi["month_decomp"][nb_idx][1]-1
            hdd1 = (HDD_real(date_p, date_s, nb_months, sl_calib, hp, HDD_date_l, date_o, Z_m, Z_m_d, range_gpu, range_gpu_inv, gpu_ind) - HDD_price_l[nb_idx])**2
            hdd2 = (HDDO_real (HDDO_price_l[nb_idx][0], date_p, date_s, nb_months, sl_calib, hp, HDD_date_l, date_o, 
                               Z_m, Z_m_d, range_gpu, range_gpum1, range_gpu_inv, gpu_ind) - HDDO_price_l[nb_idx][1])**2

            return hdd1 + hdd2

        p = openopt.NLP ( opt_fct, sl_init, lb=[0., -inf] )
        sl = p.solve ('scipy_cobyla').xf
        sl_calib[nb_idx] = sl

    return sl_calib

def HDD_payoff(T_sm):
    return np.average (np.sum (np.maximum (65. - T_sm , 0.), axis = 1))

# reduction kernel, sums the elemnts in a vector 
average_reduction = pycuda.reduction.ReductionKernel(np.dtype(np.float32), neutral="0",
                                                     reduce_expr="a+b", 
                                                     map_expr="x[i]",
                                                     arguments="float *x")


# same as above, just computes a bunch of stuff on the device 
def HDD_payoff_d (T_sm_d):
    T_sm_tmp_d = 65. - T_sm_d
    maximum_cuda ( T_sm_tmp_d ) # this changes T_sm_d, which souldnt
    rowsum_cuda_notransfer(T_sm_tmp_d, rowsum_vec_d) # writes in rowsum_vec_d

    # the reason why this is better is that it only carries over one number
    # implements return np.average( rowsum_vec_d.get() )    
    return average_reduction(rowsum_vec_d).get() / rowsum_vec_d.shape[0]


def HDD_d(t_0,t_step, N_step, T_0, sp, hp, Z_m_d, 
          range_gpu, range_gpum1, 
          range_gpu_inv, date_p, date_o):
    ttdp = (date_p - date_o).days / 365.25
    t_v_d = (t_0 + t_step * range_gpu_inv).astype(np.float32)
    T_m_v_d = T_m_d ( (ttdp + t_v_d).astype(np.float32), hp)
    T_m_d_d = T_m_der_d ( (ttdp + t_v_d).astype(np.float32),hp)

    T_sm_d = T_sim_inn_d (T_m_v_d, T_m_d_d, t_step, N_step, sp, hp, Z_m_d, 
                          range_gpu, range_gpum1, 
                          date_p, date_o) # T simulated matrix 
    return HDD_payoff_d(T_sm_d)



# 
def swap(t_0, t_step, N_step, N_step_swap, T_0, sp, hp, Z_m):
    ttdp = (date_p - date_o).days / 365.25
    t_v = t_0 + t_step * np.arange(N_step) 
    T_m_v = T_m (ttdp + t_v, hp)
    T_m_d = T_m_der (ttdp + t_v,hp) # vector as well

    T_sm = T_sim_inn (T_m_v, T_m_d, t_step, N_step_swap, sp, hp, Z_m[:,:N_step_swap]) # T simulated matrix 
    return swap_payoff (T_sm) 


def swap_payoff (T_sm):
    return np.average (np.average (T_sm , axis = 1) )



# K ... HDD strike 
def HDDO (K, t_0, t_step, N_step, T_0, sp, hp, Z_m, cp_ind = "c"):
    ttdp = (date_p - date_o).days / 365.25
    t_v = t_0 + t_step * np.arange(N_step) 
    T_m_v = T_m (ttdp + t_v, hp)
    T_m_d = T_m_der (ttdp + t_v,hp) # vector as well

    T_sm = T_sim_inn (T_m_v, T_m_d, t_step, N_step, sp, hp, Z_m) # T simulated matrix 
    return HDDO_payoff (K, T_sm, cp_ind) 

def HDDO_payoff(K, T_sm, cp_ind = "c"):
    return np.average (np.maximum (np.sum (np.maximum (65. - T_sm , 0.), axis = 1) - K, 0.) )


# writes the vector v in col n of matrix m 
# nb_sims is the number of rows (simulations in rows)
# nb_cols ... number of columns 
write_vec_in_mat_col_code = """
__global__ void write_vec_in_mat_col (float *v, float *m, int n, int nb_sims, int nb_cols) {
  
  int elt_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (elt_idx < nb_sims ) 
    m[ n  + elt_idx * nb_cols ] = v[elt_idx];
}
"""
wohdd_module = config.SourceModule(write_vec_in_mat_col_code)
wohdd_f = wohdd_module.get_function ("write_vec_in_mat_col") 
def write_vec_in_mat_col ( rowsum_vec_d, hdd_sim_d, n):
    """
    implements the following: hdd_sim[:,n] = rowsum_vec_d
    """

    nb_sims = hdd_sim_d.shape[0] # nb. rows, simulations in rows 
    nb_days = hdd_sim_d.shape[1] # nb. cols, days are in cols

    wohdd_f (rowsum_vec_d, hdd_sim_d, np.int32(n), np.int32(nb_sims), np.int32(nb_days),
             block=(nb_sims / 65535 +1,1,1), grid= (65535,1) )

# HDD option real pricing 
def HDDO_real (K, date_p, date_s, nb_months, sp_l, hp, HDD_date_l, date_o, 
               Z_m, Z_m_d, range_gpu, range_gpum1, range_gpu_inv, cp_ind = "c", gpu_ind = False):
    
    mi = months_index (HDD_date_l, sp_l[0]) # just need month_decom 
    mi_dec = month_decomp (mi["month_decomp"])

    sim_size = Z_m.shape[0]

    hdd_sim = np.zeros ((sim_size, nb_months))
    hdd_sim_d = gpa.zeros((sim_size, nb_months), dtype=np.float32)
    hdd_sim_sum_d = gpa.zeros (sim_size, dtype=np.float32)



    for n in arange (nb_months):
        t_month_start = add_months (date_s, n) 
        t_month_start_num = (t_month_start - date_p).days / 365.
        #T_0 = T_m_real (t_month_start, date_o, hp)
        t_step = 1. / 365.
        #N_step = (add_months(t_month_start, 1) - t_month_start).days 
        # COMPLETELY WRONG WRONG WRONG, PREVIOUS LINE IS CORRECT 
        N_step = 31 
        sp = sp_l[month_into_sigma(n,mi_dec)] # mapping into sp_l[ n
        ttdp = (date_p - date_o).days / 365.25
        if gpu_ind == False:
            #Z_m = np.random.normal(size=(sim_size,N_step))
            t_v = t_month_start_num + t_step * np.arange(N_step) 
            T_m_v = T_m (ttdp + t_v, hp)
            T_m_d1 = T_m_der (ttdp + t_v,hp) # vector as well

            T_sim = T_sim_inn (T_m_v, T_m_d1,  t_step, N_step, sp, hp, Z_m, date_p, date_o) 
            hdd_sim[:,n] = np.sum (np.maximum (65. - T_sim , 0.), axis = 1)

        else:
            t_v_d = (t_month_start_num + t_step * range_gpu_inv).astype(np.float32)
            T_m_v_d = T_m_d ( (ttdp + t_v_d).astype(np.float32), hp)
            T_m_d_d = T_m_der_d ( (ttdp + t_v_d).astype(np.float32),hp)

            T_sim_d = T_sim_inn_d ( T_m_v_d, T_m_d_d, t_step, N_step, sp, hp, Z_m_d, 
                                   range_gpu, range_gpum1, 
                                   date_p, date_o)

            T_sim_d = 65. - T_sim_d # obviously works, bravo pycuda
            maximum_cuda(T_sim_d)
            rowsum_cuda_notransfer (T_sim_d, rowsum_vec_d)
            write_vec_in_mat_col ( rowsum_vec_d, hdd_sim_d, n) # impl. : hdd_sim_d[:,n] = rowsum_vec_d
        
        if gpu_ind == False:
            return np.average (np.maximum ( np.sum (hdd_sim, axis =1 ) - K, 0.))

        else:
            rowsum_cuda_notransfer(hdd_sim_d, hdd_sim_sum_d)
            hdd_sim_sum_d = hdd_sim_sum_d - K
            maximum_cuda (hdd_sim_sum_d)

            #print "opt_d = ", hdd_sim_sum_d

            # this works for now 
            #return np.average (np.maximum ( np.sum (hdd_sim, axis =1 ) - K, 0.))
            return average_reduction (hdd_sim_sum_d ).get() / hdd_sim_sum_d.shape[0]



def HDDO_histo (K, date_p, date_s, nb_months, hp, HDD_date_l, date_o, 
                Z_m , Z_m_d, range_gpu, range_gpum1, range_gpu_inv, cp_ind = 'c', gpu_ind = False):
    return HDDO_real (K, date_p, date_s, nb_months, [(0.,0.)] * len (HDD_date_l), hp, HDD_date_l, date_o, Z_m, Z_m_d, range_gpu, range_gpum1, range_gpu_inv, cp_ind, gpu_ind)



# same as above, just that most operations are performed on the device
def HDDO_payoff_d (K, T_sm_d, cp_ind = "c"):
    T_sm_tmp_d = 65. - T_sm_d
    maximum_cuda(T_sm_tmp_d)
    HDD_payoff_tmp = cumsum_cuda(T_sm_tmp_d) - K

    # implements: return np.average (np.maximum (HDD_payoff_tmp.get(), 0.) )
    return average_reduction (maximum_cuda (HDD_payoff_tmp)).get / HDD_payoff_tmp.shape[0]


# def HDDO_d(K, t_0, t_step, N_step, T_0, sp, hp, Z_m_d, range_gpu, cp_ind = "c"):
#     ttdp = (date_p - date_o).days / 365.25
#     t_v_d = (t_0 + t_step * range_gpu_inv).astype(np.float32)
#     T_m_v_d = T_m_d ( (ttdp + t_v_d).astype(np.float32), hp)
#     T_m_d_d = T_m_der_d ( (ttdp + t_v_d).astype(np.float32),hp)

#     T_sm_d = T_sim_inn_d (T_m_v_d, T_m_d_d, t_step, N_step, sp, hp, Z_m_d, 
#                           range_gpu, date_o, date_p) # T simulated matrix 
    
#     return HDDO_payoff_d (K, T_sm_d, cp_ind) 


# calibration of both HDD and HDD option 
# targets ar HDD_tar and HDDO_tar
def HDD_calib (swap_tar, HDD_tar, HDDO_tar, K, t_0, t_step, N_step, T_0, sp, hp, Z_m):

    def opt_f (sigma_lam):
        sp_n = (sigma_lam[0], sigma_lam[1]) 
        T_sm = T_sim_inn (t_0,t_step, N_step, T_0, sp_n, hp, Z_m)
        f0 = 0. #(swap_payoff(T_sm) - swap_tar)**2
        f1 = (HDD_payoff(T_sm) - HDD_tar)**2 
        f2 = (HDD_opt_payoff (K, T_sm) - HDDO_tar)**2 

        return 0.92 * f0 + 0.04 * f1 + 0.04 * f2

    sol = fmin (opt_f, [0.2, 0.5])
    print "sol = ", opt_f(sol)
    print "sol par = ", sol
    return sol

# device calibration
# def HDD_calib_d (swap_tar, HDD_tar, HDDO_tar, K, t_0, t_step, N_step, T_0, sp, hp, Z_m, 
#                  Z_m_d, range_gpu):

#     def opt_f (sigma_lam):
#         sp_n = (sigma_lam[0], sigma_lam[1]) 
#         LLL
#         ttdp = (date_p - date_o).days / 365.25
#         t_v_d = (t_0 + t_step * range_gpu_inv).astype(np.float32)
#         T_m_v_d = T_m_d ( (ttdp + t_v_d).astype(np.float32), hp)
#         T_m_d_d = T_m_der_d ( (ttdp + t_v_d).astype(np.float32),hp)

#         T_sm_d = T_sim_inn_d (t_0,t_step, N_step, T_0, sp_n, hp, Z_m_d, range_gpu)
#         f0 = 0. #(swap (t_0, t_step, 7, T_0, sp_n, hp, Z_m) - swap_tar)**2
#         f1 = (HDD_payoff_d(T_sm_d) - HDD_tar)**2 
#         f2 = (HDD_opt_payoff_d(K, T_sm_d) - HDDO_tar)**2 

#         return 0.92 * f0 + 0.04 * f1 + 0.04 * f2

#     sol = fmin (opt_f, [0.2, 0.5])
#     print "sol = ", opt_f(sol)
#     print "sol par = ", sol
#     return sol

    


def T_par_inn ( T_m_v, T_m_d, dt, sp, hp, Z_m, date_p, date_o):
    """
    fast simulation of the weather model partial innovations  
    """
    a = hp[5]
    sigma, sigma_lam = sp
    
    t3_c = time.time()
    res = (T_m_d + a * T_m_v - sigma_lam ) * dt + sigma * sqrt (dt) * Z_m
    t3_c = time.time() - t3_c

    #print "cpu t = ", (t3_c)

    return res


# Z_m_d NEEDS to have 0's in the first column
def T_par_inn_d ( T_m_v_d, T_m_d_d, dt, sp, hp, Z_m_d1, date_p, date_o):
    """
    same as T_par_inn, except that it runs on GPU device 
    """

    #t1_d = time.time()
    #t2_d = time.time()
    a = hp[5]
    sigma, sigma_lam = sp
    #t2_d = time.time() - t2_d

    # this implements ( *a _has_ be on the right )
    #t3_d = time.time()
    v_d = (T_m_d_d + T_m_v_d * a - sigma_lam ) * dt
    #t3_d = time.time() - t3_d
    
    # v = np.append ( T_0, ( T_m_d + a * T_m_v  - sigma_lam ) * dt )

    #t4_d = time.time()
    inn_d = Z_m_d1 * (sigma * sqrt (dt)) # mult. _has_ to be on the right (problems with pycuda)
    #t4_d = time.time() - t4_d

    #t5_d = time.time()
    vtpm ( v_d, inn_d, 'p' ) # inn_d <- v_d + inn_d 
    #t5_d = time.time() - t5_d
    #t1_d = time.time() - t1_d
    #print "gpu t = ", (t1_d, t2_d, t3_d, t4_d, t5_d)

    return inn_d # inn_d



# simulation from partial innovations, FAST simulation 
def T_sim_inn (T_m_v, T_m_d, t_step, N_step, sp, hp, Z_m, date_p, date_o):
    #t_v = t_0 + t_step * np.arange(N_step) 
    T_s_1 = T_par_inn (T_m_v, T_m_d, t_step, sp, hp, Z_m, date_p, date_o)
    
    # THERE SHOULD BE T_0, perhaps we can circumvent this 
    #T_s_1 = np.append (T_0 * np.ones((Z_m.shape[0],1)), T_s_1, axis=1)
    a = hp[5]
    T_s_2 = T_s_1 * (1-a * t_step)**arange(N_step-1,-1,-1)

    return np.cumsum(T_s_2,axis=1) / (1-a*t_step)**arange(N_step-1,-1,-1)


# same as above, just done on the device 
def T_sim_inn_d (T_m_v_d, T_m_d_d, t_step, N_step, sp, hp, Z_m_d1, 
                 range_gpu, range_gpum1, 
                 date_p, date_o):
    # this below is inlined directly below 
    # T_s_1_d = T_par_inn_d ( T_m_v_d, T_m_d_d, t_step, sp, hp, Z_m_d, date_p, date_o)
    
    a = hp[5]
    sigma, sigma_lam = sp
    # this implements ( *a _has_ be on the right )
    v_d = (T_m_d_d + T_m_v_d * a - sigma_lam ) * t_step
    T_s_1_d = Z_m_d1 * (sigma * sqrt (t_step)) # mult. _has_ to be on the right (problems with pycuda)
    vtpm ( v_d, T_s_1_d, 'p' ) # inn_d <- v_d + inn_d 

    # appending from T_sim_inn is done in T_par_inn_d directly 
    # a = hp[5]
    # implements: T_s_1_d =  (1-a * t_step)**arange(N_step,-1,-1) * T_s_1_d 
    # TO IMPROVE BELOW:
    # N_step = 31 in this case, otherwise can be different
    # mult2 = gpa.to_gpu( (1-a * t_step)**(- arange(N_step, -1,-1)).astype(np.float32) )
    range_gpu_l = range_gpu.apow( 1. - a * t_step )
    range_gpum1_l = range_gpum1.apow(1. - a * 1./365.)
    vtpm ( range_gpu_l, T_s_1_d, 't' ) # multiply

    # following 2 statements implement the following 
    # np.cumsum(T_s_1_d.get(),axis=1)/ (1-a*t_step)**arange(N_step,-1,-1)
    cumsum_cuda (T_s_1_d) # cumsum on cuda
    #vtpm ( pow (range_gpu, -1.), T_s_1_d, 't')
    vtpm ( range_gpum1_l, T_s_1_d, 't')

    return T_s_1_d

