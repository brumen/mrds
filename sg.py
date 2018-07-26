import numpy as np
from scipy.misc import comb
import quad1d  # my own 1d quadrature only for sparse grids
import opd.opd_avx as opd_avx

# 
# My own sparse grid implementation
#
# Abbreviations: 
#   sparse_grid abbreviated with sg
#   points ... p
#   weights ... w


def index_grid(K, l):
    """
    constructs all K- length vectors of integers 1... which sum to less than N
    K ... dimension of the vectors
    l ... level
    # constructs the index grid for sparse grid and saves it into Acc
    # k ... helper index that counts on what level we are
    # R ... current list
    # Acc_ind ... accumulator of indices
    # Acc_w ... accumulator of weights
    # N_low ... lower level of the grid
    # N_up ... upper level
    """
    N_low = l
    N_up = l + K - 1

    def index_grid_help(k, R, K, N_low, N_up, Acc_ind, Acc_w):
        if k == K:
            R_sum = int(np.sum(R))
            Acc_ind.append(R)
            Acc_w.append((-1)**(N_up - R_sum) * comb(K - 1, R_sum - N_low))  # comb (K - 1, R_sum-N_low ) )
        elif k == (K-1):
            R_sum = int(np.sum(R))
            for ind in range(max(N_low-R_sum, 0), N_up + 1 - R_sum):  # special case on the last k_d
                index_grid_help(k+1, R + [ind], K, N_low, N_up, Acc_ind, Acc_w)
        else:
            R_sum = int(np.sum(R))
            for ind in range(0, N_up + 1 - R_sum):  # !!!!! Used to be 0 and N-Rsum
                index_grid_help(k+1, R + [ind], K, N_low, N_up, Acc_ind, Acc_w)

    Acc_ind = []
    Acc_w = [] 
    index_grid_help(0, [], K, N_low, N_up, Acc_ind, Acc_w)
    return [Acc_ind, Acc_w] 


def pairs_final(vec_list):
    """
    does all pairs from a list of multiple vectors
    (is separate to be used by fem2d)
    """
    def pairs(vec_list_mod):
        vec_list_mod_len = len(vec_list_mod)
        
        def outer_vec(v1, v2):
            return [v1[i] + v2[j] for i in range(len(v1))
                    for j in range(len(v2))]
        
        if vec_list_mod_len is 1:
            return vec_list_mod[0]
        else:
            return outer_vec(vec_list_mod[0], pairs(vec_list_mod[1:]))

    # modified vector list 
    vec_list_mod = map(lambda y: map(lambda x: [x], y), vec_list)
    return pairs(vec_list_mod)


def sg_p(D, l, one_d_discret='gauss_hermite', one_d_grid=[]):
    """
    construct sparse grid points
      D ... dimension
      level ... sparse grid level
      one_d_discret ... 1-D discretization of points
        gauss_hermite ... G-H discretization
        linear ... equidistantly  spaced
        manual ... suplied in one_d_grid
    """
    ig = index_grid(D, l)[0]
    if one_d_discret is 'gauss_hermite':
        hg = [[list(quad1d.gh_pw_hints(n)[0]) for n in B] for B in ig]
    elif one_d_discret is 'linear':
        hg = [[list(np.arange(-2**n, 2**n+1)/np.double(2**n)) for n in B] for B in ig]
    elif one_d_discret is 'manual':
        hg = [[list(one_d_grid)] * len(B) for B in ig]

    spgb = [pairs_final(y) for y in hg]
    # flattening, this is really effective !!!! IS THIS REALLY NEEDED BELOW 
    spg = []
    for par_list in spgb:
        for inv_list in par_list:
            spg.append(inv_list)

    return spg


def sg_w(D, l, one_d_discret='gauss_hermite'):
    """
    constructs weights for sparse grid of dimension D and level l
      D ... dimension
      l ... level
    """
    iwg = index_grid(D, l)  # index and weight grid
    ig = iwg[0]  # index grid
    wg = iwg[1]  # weight grid

    if one_d_discret is 'gauss_hermite':
        hg = [[list(quad1d.gh_pw_hints(n)[1]) for n in B] for B in ig]
    elif one_d_discret is 'linear':  # weights are all of 1
        hg = [[list(np.repeat(np.double(2**(-n)), 2**(n+1)+1)) for n in B] for B in ig]

    spgb = zip([pairs_final(y) for y in hg], wg)

    # flattening, this is really effective
    spg = []
    for par_list, w_list in spgb:
        for inv_list in par_list:
            spg.append([inv_list, w_list])

    # mapping the product
    return [np.prod(x[0]) * x[1] for x in spg]


def sg_quad(D, l, f, one_d_discret='gauss_hermite',
            xmm_use=True):
    """
    sparse grid quadrature
    integrates \int f(x) e^(-x**2) / (2 * pi)^(D/2)
      D ... dimension of the function
      l ... level
      f ... function to integrate, takes a vector, i.e. f(x) = f(x[0], x[1]...)
      one_d_discret ... 1 dimen. discretization, can be either
        gauss_hermite (support -inf, inf)
        linear (support [-1, 1] )
    """
    sqrt2 = np.sqrt(2.)
    sqrt_pi = np.sqrt(np.pi)
    g = lambda x: f(x * sqrt2) / sqrt_pi**D
    weights = np.array(sg_w(D, l, one_d_discret))
    vals = np.array(map(g, [np.array(x) for x in sg_p(D, l, one_d_discret)])).flatten()
    if not xmm_use:
        return np.sum(weights*vals)
    else:
        return opd_avx.num_quad(weights, vals, len(vals))
