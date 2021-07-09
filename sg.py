import numpy as np
from scipy.special import comb
from typing        import Tuple, List, Callable

import mrds.quad1d as quad1d
from mrds.tolling.opd import opd_avx


#
# Abbreviations:
#   sparse_grid abbreviated with sg
#   points ... network_struct
#   weights ... w


# TODO: WRITE THIS RECURSION NICELY.
def index_grid_help(k, R, K, N_low, N_up, Acc_ind, Acc_w):
    if k == K:
        R_sum = int(np.sum(R))
        Acc_ind.append(R)
        Acc_w.append((-1) ** (N_up - R_sum) * comb(K - 1, R_sum - N_low))  # comb (K - 1, R_sum-N_low ) )
    elif k == (K - 1):
        R_sum = int(np.sum(R))
        for ind in range(max(N_low - R_sum, 0), N_up + 1 - R_sum):  # special case on the last k_d
            index_grid_help(k + 1, R + [ind], K, N_low, N_up, Acc_ind, Acc_w)
    else:
        R_sum = int(np.sum(R))
        for ind in range(0, N_up + 1 - R_sum):  # !!!!! Used to be 0 and N-Rsum
            index_grid_help(k + 1, R + [ind], K, N_low, N_up, Acc_ind, Acc_w)


def index_grid(K : int , l : int) -> Tuple[List, List]:
    """  Constructs all K- length vectors of integers 1... which sum to less than N

    :param K: dimension of the vectors
    :param l: level of sparse grid
    :returns: tuple of accumulated indices, and accumulated weights.
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

    Acc_ind = []
    Acc_w = [] 

    index_grid_help(0, [], K, N_low, N_up, Acc_ind, Acc_w)

    return Acc_ind, Acc_w


def pairs_final(vec_list):
    """ Does all pairs from a list of multiple vectors (is separate to be used by fem2d).

    """

    def pairs(vec_list_mod):
        vec_list_mod_len = len(vec_list_mod)

        if vec_list_mod_len is 1:
            return vec_list_mod[0]

        v1 = vec_list_mod[0]
        v2 = pairs(vec_list_mod[1:])

        return [ v1[i] + v2[j]
                 for i in range(len(v1))
                 for j in range(len(v2)) ]

    return pairs(map(lambda y: map(lambda x: [x], y), vec_list))


def sg_p(D : int, l : int, one_d_grid : List[float], one_d_discret : str ='gauss_hermite' ):
    """ Construct sparse grid points:

    :param D: dimension of sparse grid
    :param l: sparse grid level
    :param one_d_grid: one dimensional grid given
    :param one_d_discret:  grid used for 1-D discretization of points
        gauss_hermite ... G-H discretization
        linear ... equidistantly  spaced
        manual ... suplied in one_d_grid
    :returns:
    """

    ig, _ = index_grid(D, l)

    if one_d_discret is 'gauss_hermite':
        hg = [[list(quad1d.gh_pw(n)[0]) for n in B] for B in ig]

    elif one_d_discret is 'linear':
        hg = [[list(np.arange(-2**n, 2**n+1)/np.double(2**n)) for n in B] for B in ig]

    elif one_d_discret is 'manual':
        hg = [[list(one_d_grid)] * len(B) for B in ig]

    else:
        raise RuntimeError('one_d_discret parameter not one of gauss_hermite, linear, manual')

    # flattening, this is really effective !!!!
    return [inv_list
            for par_list in [pairs_final(y) for y in hg]
                for inv_list in par_list ]


def sg_w(D : int, l : int, one_d_discret='gauss_hermite') -> List[float]:
    """ Constructs weights for sparse grid of dimension D and level l

    :param D: dimension of sparse grid.
    :param l: level of sparse grid.
    :param one_d_discret: type of 1-dimensional discretization.
    """

    ig, wg = index_grid(D, l)  # index and weight grid

    if one_d_discret is 'gauss_hermite':
        hg = [[list(quad1d.gh_pw(n)[1]) for n in B] for B in ig]
    elif one_d_discret is 'linear':  # weights are all of 1
        hg = [[list(np.repeat(np.double(2**(-n)), 2**(n+1)+1)) for n in B] for B in ig]
    else:
        raise RuntimeError('one_d_discret not one of gauss_hermite, linear')

    # sparse grid
    spg = [ [inv_list, w_list]
            for par_list, w_list in zip([pairs_final(y) for y in hg], wg)
                for inv_list in par_list ]

    # mapping the product
    return [np.prod(x[0]) * x[1] for x in spg]


def sg_quad(D : int, l : int, f : Callable, one_d_discret : str = 'gauss_hermite', xmm_use : bool = True) -> float:
    """ Sparse grid quadrature of the function f.
        Integrates \int f(x) e^(-x**2) / (2 * pi)^(D/2)

    :param D: dimension of function f
    :param l: level of sparse grid.
    :param f: function to integration, takes a vector, i.e. f(x) = f(x[0], x[1]...)
    :param one_d_discret: 1 dimen. discretization, can be either
        gauss_hermite (support -inf, inf)
        linear (support [-1, 1] )
    :param xmm_use: indicator whether to use xmm optimization.
    """

    sqrt2 = np.sqrt(2.)
    sqrt_pi = np.sqrt(np.pi)
    g = lambda x: f(x * sqrt2) / sqrt_pi**D
    weights = np.array(sg_w(D, l, one_d_discret))
    vals = np.array(map(g, [np.array(x) for x in sg_p(D, l, [], one_d_discret)])).flatten()

    if xmm_use:
        return opd_avx.num_quad(weights, vals, len(vals))

    return np.sum(weights*vals)
