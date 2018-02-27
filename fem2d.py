# 2D FEM Method
# 

# File defines:
import ms.version
ms.version.addpkg ('numpy', '1.4.1')
from numpy import *
ms.version.addpkg ('scipy', '0.7.2')
import scipy
import scipy.optimize
import scipy.integrate
import scipy.special
import scipy.stats 
import scipy.optimize 

# CHECK WHICH OF THESE YOU REALLY NEED 
import scipy.sparse 
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand

# iterators
import itertools 

# for pairs function we need this 
from sg import pairs_final

# market object class 
class mo:
    h = 1.0/12.0 # number of space discretizations in every dimension
    delta_t = 0.001 # time step 
    K = 1
    sigma1 = 0.25
    sigma2 = 0.25
    rho = 0.3
    r = 0.05 # interest rate
    scaling = 1 # 1... scaled delta by 1/h , 0... no scaling 

class moN:
    h = 1.0/12.0 # number of space discretizations in every dimension
    delta_t = 0.001 # time step 
    K = 1
    sigma_vec = array([0.25, 0.3, 0.35])
    rho_mat = array([[1, 0.2, 0.4], [0.2, 1, 0.5], [0.4, 0.5, 1]])
    r = 0.05 # interest rate
    scaling = 1 # 1... scaled delta by 1/h , 0... no scaling 


def fem2d_time_step ():
    # time stepping 
    for t in mo.delta * arange (1/mo.delta_t): # 0:mo.delta_t:1 
        V_delta = dot (V, delta) # right side matrix 
        delta_V = dot (delta, V) # rhs 
  
        RHS = - 0.5 * mo.sigma1 ** 2 * dot (delta_p.T , V_delta) - 0.5 * mo.sigma2**2 * dot (delta_V, delta_p) \
              - 0.5 * mo.sigma1**2 * dot (delta_2p , V_delta) - 0.5 * mo.sigma2**2 * dot (delta_V , delta_2p) + \
              mo.rho * mo.sigma1 * mo.sigma2 * dot (delta_p.T , V * delta_p) + \
              ( 1/mo.delta_t - mo.r) * dot (delta * V_delta)
  
        V = linalg.solve (delta, linalg.solve ( (dot (delta, mo.delta_t).T , RHS.T)).T ) # matlab version : delta \ RHS / delta * mo.delta_t 

    # MISSING MISSING MISSING 
    return V


# h-FEM time step 
# u0_ten ... tensor of initial u0
def hfem_ts (u0_ten, moN, x_vec_l, h_vec_l):
    # time stepping 
    V = u0_ten
    for t in mo.delta * arange (1/mo.delta_t): # 0:mo.delta_t:1 
        U, Us, Uss, Usr = fk_ten (N, V, x_vec_l, h_vec_l)
        V = V + moN.delta_t * ( -0.5 * moN.sigma_vec * (Us + Uss) + moN.rho_mat ) # r part MISSING  
         
    return V

# generates the matrix Delta (tridiagonal matrix for the
# dimension of the matrix 
# mo ... (market object), really just params 
def delta_mat (N, mo):
    
    one_to_nminus_range = arange (N-1)
    
    # mass matrix M
    delta = scipy.sparse.lil_matrix((N, N))
    delta.setdiag ( ones (N) * (4.0 / 6.0 ) )
    delta[1 + one_to_nminus_range, one_to_nminus_range] = 1.0 / 6.0
    delta[one_to_nminus_range, 1 + one_to_nminus_range] = 1.0 / 6.0
    delta[0,0] = 2.0 / 6.0 # NOT 4
    delta[N-1,N-1] = 2.0 /6.0 

    # non-symmetric cross matrix C
    delta_p = scipy.sparse.lil_matrix ((N,N))
    delta_p[ 1+one_to_nminus_range, one_to_nminus_range] = - 0.5 / mo.h # below diagonal 
    delta_p[ one_to_nminus_range, 1 + one_to_nminus_range ] = 0.5 / mo.h # above diagonal 
    delta_p[0,0] = -0.5 / mo.h
    delta_p[N-1,N-1] = 0.5 / mo.h

    # stiffnes matrix S
    delta_2p = scipy.sparse.lil_matrix ((N,N))
    delta_2p[ 1+one_to_nminus_range, one_to_nminus_range] = - 1 / mo.h ** 2 # below diagonal 
    delta_2p[ one_to_nminus_range, 1 + one_to_nminus_range ] = - 1 / mo.h ** 2 # above diagonal 
    delta_2p.setdiag ( ones (N) * 2 / mo.h ** 2)
    delta_2p[0,0] = 1 / mo.h ** 2
    delta_2p[N-1,N-1] = 1 / mo.h ** 2

    return [delta, delta_p, delta_2p]


def kron_n (n, M_l): # kronecker product of all matrices in M_l 
    if n == (len(M_l)-1):
        return M_l[n] 
    else: 
        return kron ( M_l[n], kron_n (n+1,M_l) )

# Feynman-Kac tensors
# N is the tensor dimension, perhaps it can be inferred 
def fk_ten (N, u_ten, x_vec_l, h_vec_l):
    u_ten = tensordot ( u_ten, Delta_ten(x_vec_l, h_vec_l), axes=range(N) )
    u_s_ten = sum ( [ tensordot (u_ten, Delta1_ten(n,x_vec_l, h_vec_l)) for n in range (N) ] )
    u_ss_ten = sum ( [ tensordot (u_ten, Delta2_ten(n,x_vec_l, h_vec_l)) for n in range (N) ] )
    u_sr_ten = sum ( [ tensordot (u_ten, Delta11_ten(i,j,x_vec_l, h_vec_l)) for i in range (N) \
                      for j in range (i) ] )
    return [u_ten, u_s_ten, u_ss_ten, u_sr_ten] 

# constructs the tensor product \sum_j u(j) A(j,k)
# THIS HAS TO BE CORRECTED 
def u_tilde_constr (k, N, J, j, u_tilde, u, A):
    if k == N:
        u_tilde[j] = sum (u * A[:,j])
    else:
        for j_cur in range ( J[k] ):
            u_tilde_constr (k+1, N, J, j.append(j_cur), u_tilde, u, A) 


# constructs a tensor from a function
# params ... extra parameter for function u_fct = u_fct (x, params)
# u_fct ... = u_fct (x, params)
def u_ten_constr (N, u_fct, x_vec_l, params):
    x_ten = pairs_final (x_vec_l) # vector of abscisses, x_vectors of all pairs 
    dim_lens = [ len(x_vec_l[idx]) for idx in arange(len(x_vec_l)) ] # dimension sizes for x_vec_l 
    idx_ten = pairs_final ([ arange(len(x_vec_l[idx])) for idx in arange(len(x_vec_l)) ] ) # index tensor 
    u_ten = zeros (dim_lens)
    for idx, idx_nb in zip (idx_ten, range(len(idx_ten))):
        u_ten[tuple(idx)] = u_fct(x_ten[idx_nb], params)
    return u_ten 



# constructs the Delta tensor Delta (i,j), where i=(i_1,i_2, ... i_N), j=(j_1, j_2, ... j_N)
# IMPROVE IMPROVE - kron does not handle sparse matrices correctly - .toarray() needed
def Delta_ten (x_vec_l, h_vec_l):
    if x_vec_l == []:
        return array([1])
    else: 
        mm_l = [ compute_mass_matrix (x_vec_l[ind], h_vec_l[ind]).toarray() for ind in arange (len(x_vec_l))] # list of mass matrices
        #print mm_l
        return kron_n (0, mm_l)

# constructs the tensor Delta^1 (K=0, ... N-1)
def Delta1_ten (K, x_vec_l, h_vec_l):
    return kron ( kron (Delta_ten(x_vec_l[0:K], h_vec_l[0:K]), \
                        compute_cross_matrix(x_vec_l[K],h_vec_l[K]).toarray() ), \
                 Delta_ten (x_vec_l[(K+1):len(x_vec_l)], h_vec_l[(K+1):len(h_vec_l)]) )

# constructs the tensor Delta^1 (K<M=0, ... N-1)
def Delta11_ten (K, M, x_vec_l, h_vec_l):
    return kron (kron ( Delta1_ten (K, x_vec_l[0:M], h_vec_l[0:M]), \
                        compute_cross_matrix(x_vec_l[M],h_vec_l[M]).toarray() ), \
                 Delta_ten (x_vec_l[(M+1):len(x_vec_l)], h_vec_l[(M+1):len(h_vec_l)]) )

# constructs the tensor Delta^1 (K=0, ... N-1)
def Delta2_ten (K, x_vec_l, h_vec_l):
    return kron ( kron (Delta_ten(x_vec_l[0:K], h_vec_l[0:K]), \
                        compute_stiffness_matrix(x_vec_l[K],h_vec_l[K]).toarray() ), \
                 Delta_ten (x_vec_l[(K+1):len(x_vec_l)], h_vec_l[(K+1):len(h_vec_l)]) )


# u is a tensor of dimension n_dim (= len (j_vec) )
# N ... where we are in recursion 
def u_tensor(N, u, j_vec, x_vec_l, h_vec_l):
    n_dim = len (j_vec)
    x_lens = [ len(x_vec_l[ind]) for ind in len (x_vec_l)]
    # recursive call 
    


# computes the sparse MASS  matrix of h-FEM 
def compute_mass_matrix (x_vec, h_vec):
    # computes the integral STATE WHICH INTEGRAL 
    def compute_mass_integral (x_i, h_i, x_j, h_j, A, B, sign_1, sign_2):
        return B-A + sign_1 * ( (B-x_i)**2 - (A-x_i)**2 ) / h_i + \
               sign_2 * ( (B-x_j)**2 - (A-x_j)**2 ) / h_j + \
               sign_1 * sign_2 * 4.0 / (h_i * h_j ) * (B**3 / 3.0 - (x_i + x_j) * B**2 / 2.0 + x_i * x_j * B  \
                                                       - A**3 / 3.0 + (x_i + x_j) * A**2 / 2.0 - x_i * x_j * A )

    return compute_matrix_pos_only (compute_mass_integral, x_vec, h_vec)


# CROSS MATRIX 
def compute_cross_matrix (x_vec, h_vec):

    # computes the integral STATE WHICH INTEGRAL 
    def compute_cross_integral (x_i, h_i, x_j, h_j, A, B, sign_1, sign_2):
        return sign_1 * 2.0 / h_i * (B-A) + sign_1 * sign_2 * 2.0 / (h_i * h_j ) * \
               ( (B-x_j)**2 - (A-x_j)**2 ) 

    return compute_matrix_pos_only(compute_cross_integral, x_vec, h_vec)


# STIFFNESS MATRIX 
def compute_stiffness_matrix (x_vec, h_vec):

    # computes the integral STATE WHICH INTEGRAL 
    def compute_stiff_integral (x_i, h_i, x_j, h_j, A, B, sign_1, sign_2):
        return  sign_1 * sign_2 * 4.0 / (h_i * h_j ) * (B-A)

    return compute_matrix_pos_only(compute_stiff_integral, x_vec, h_vec)



# constructs a sparse matrix with positive only elements 
def compute_matrix_pos_only(fct, x_vec, h_vec):
    mat = scipy.sparse.lil_matrix((len (x_vec), len(x_vec)))

    for i in range ( len (x_vec) ):
        for j in range ( len (x_vec ) ):
            mat_tmp = compute_matrix_element_seq (fct, x_vec[i], h_vec[i], x_vec[j], h_vec[j] )
            #print "Mat tmp = ", mat_tmp
            if ( mat_tmp > 0):
                mat[i,j] = mat_tmp

    return mat 

def compute_matrix_element_seq (fct, x_i, h_i, x_j, h_j ):
    l_i = x_i - h_i / 2.0
    r_i = x_i + h_i / 2.0
    l_j = x_j - h_j / 2.0
    r_j = x_j + h_j / 2.0

    return compute_matrix_element_seq_interval (fct, l_i, x_i, l_j, x_j, 1, 1 ) + \
           compute_matrix_element_seq_interval (fct, l_i, x_i, x_j, r_j, 1, -1 ) + \
           compute_matrix_element_seq_interval (fct, x_i, r_i, l_j, x_j, -1, 1 ) + \
           compute_matrix_element_seq_interval (fct, x_i, r_i, x_j, r_j, -1, -1 ) 

# sequential version of one element 
def compute_matrix_element_seq_interval (fct, l_i, r_i, l_j, r_j, sign_1, sign_2 ):
    x_i = (l_i + r_i) / 2.0
    h_i = (r_i - l_i)
    x_j = (l_j + r_j) / 2.0
    h_j = (r_j - l_j)
    
    #print "Intervals: [", l_i,",",r_i,"]"
    #print "Intervals: [", l_j,",",r_j,"]"
    
    #if (r_j < l_i ) or (r_i < l_j ):
    #    return 0
    if (l_i <= l_j ) and ( l_j < r_i) and (r_i <= r_j):
    #    print 1
        return fct (x_i, h_i, x_j, h_j, l_j, r_i, sign_1, sign_2 )
    elif (l_i <= l_j ) and (r_j < r_i):
    #    print 2
        return fct (x_i, h_i, x_j, h_j, l_j, r_j, sign_1, sign_2 )
    elif (l_j < l_i ) and (r_i <= r_j):
    #    print 3
        return fct (x_i, h_i, x_j, h_j, l_i, r_i, sign_1, sign_2 )
    elif (l_j < l_i ) and ( l_i < r_j) and (r_j < r_i):
    #    print 4
        return fct (x_i, h_i, x_j, h_j, l_i, r_j, sign_1, sign_2 )
    else:
    #    print 5
        return 0.0

# parallel version of the function above 
def compute_matrix_element_par_interval (fct, l_i, r_i, l_j, r_j, sign_1, sign_2 ):
    x_i = (l_i + r_i) / 2.0
    h_i = (r_i - l_i)
    x_j = (l_j + r_j) / 2.0
    h_j = (r_j - l_j)

    r = ( (l_i <= l_j ) and ( l_j < r_i) and (r_i <= r_j) ) * fct (x_i, h_i, x_j, h_j, l_j, r_i, sign_1, sign_2 ) + \
        ( (l_i <= l_j ) and (r_j < r_i) ) * fct (x_i, h_i, x_j, h_j, l_j, r_j, sign_1, sign_2 ) + \
        ( (l_j < l_i ) and (r_i <= r_j) ) * fct (x_i, h_i, x_j, h_j, l_i, r_i, sign_1, sign_2 ) + \
        ( (l_j < l_i ) and ( l_i < r_j) and (r_j < r_i) ) * fct (x_i, h_i, x_j, h_j, l_i, r_j, sign_1, sign_2 )

    return r 
