# 1d integration.

import mrds.config as config
import numpy as np
from numpy import sqrt, arange, diag, prod, pi, array, maximum, kron
from scipy.linalg import eig
import math

if config.CUDA_PRESENT:
    from pycuda import gpuarray as gpa

from functools import lru_cache


@lru_cache(maxsize=100)
def gh_pw(n : int):
    """ Pre-computes the Gauss-Hermite quadrature

    abbreviation:
       gh ... Gauss-Hermite
       network_struct ... points (abscissas)
       w ... weights

    :param n: level of gauss-hermite integration
    :returns:   # abscissas, weights
    """

    if n == 0:
        return [array([0]), array([0])]

    if n == 1:
        return [array([0]), array([1.77245385])]

    # eigenvalues of A_n
    vec_gh = sqrt(arange(1, n)/2.)
    mat_gh = diag(vec_gh, 1) + diag(vec_gh, -1)
    gh_matrix_n = eig(mat_gh)[0].real  # eigenvalues (real) <- abscissas of GH

    char_poly = lambda x, n: prod(gh_matrix_n - x) * (-2.0)**n  # characteristic polynomial
    weights = (2.0**(n+1) * math.factorial(n) * sqrt(pi)) / \
        (array(map(lambda x: char_poly(x, n+1), gh_matrix_n)))**2

    return (gh_matrix_n, weights)


def gh_quad(f, n):
    """
    1 dimensional hermite quadrature 1/sqrt(2pi) * int_R e^{-x^2/2} f(x) dx
    inputs:

    :param n: order of integration
    :param f: function to integrate
    """

    sqrt2 = sqrt(2.)
    points, weights = gh_pw(n)  # points and weights

    return np.sum(weights * map(lambda x: f(x * sqrt2), points)) / sqrt(pi)


def gh_quad_2d(f, n12):
    """
    2-dim hermite quadrature: 1/(2pi) *
    """
    n1, n2 = n12
    return gh_quad(lambda x: gh_quad(lambda y: f(x, y), n2), n1)


def gh_quad_3d(f, n123):
    """
    3-dim hermite quadrature: 1/(2pi) *
    """
    n1, n2, n3 = n123
    return gh_quad(lambda x: gh_quad(lambda y: gh_quad(lambda z: f(x, y, z), n3), n2), n1)


def gh_quad_4d(f, n1234):
    """
    4-dim hermite quadrature: 1/(2pi) *
    """
    n1, n2, n3, n4 = n1234
    return gh_quad(lambda x: gh_quad(lambda y: gh_quad(lambda z: gh_quad(lambda w: f(x, y, z, w), n4),
                                                       n3),
                                     n2),
                   n1)


def gh_quad_spread_gpu (x,w, params):
    """
    same as above, just using GPU,
    spread option is coded directly in the
    """
    #x, w = gh_pw_hints(n)
    
    #params = arange (1., 3000., 0.5)
    params_d = gpa.to_gpu (params).astype(np.float32)
    x_d = gpa.to_gpu(x.astype(np.float32))
    w_d = gpa.to_gpu(w).astype(np.float32)
    res_d = gpa.to_gpu(np.zeros((len(params), len(x)))).astype(np.float32)
    # computes the spread option 
    spread_file = open(config.work_dir + "spread_fun.cu").read()
    spread_module = config.SourceModule(spread_file)
    spread_f = spread_module.get_function("spread_f")  # spread function
    block_sel = (len(x), 1, 1)
    grid_sel = (len(params), 1)
    
    spread_f(params_d, x_d, res_d, block=block_sel, grid=grid_sel)
    cublas.cublasSgemv_d(1., res_d, w_d, 0., params_d)

    return params_d.get()


def gh_impl_quad_spread_gpu(x, w, target):
    """
    same as above, tries to find implied params
    """
    #guess_d = gpa.to_gpu ( array([0]) ).astype(np.float32) # guess for target
    x_d = gpa.to_gpu(x.astype(np.float32))
    w_d = gpa.to_gpu(w).astype(np.float32)
    res_d = gpa.to_gpu(np.zeros(len(x))).astype(np.float32)
    # computes the spread option 

    spread_file = open(config.work_dir + "spread_fun.cu").read()
    spread_module = config.SourceModule(spread_file)
    spread_f_one_param = spread_module.get_function("spread_f_one_param")
    block_sel = (len(x), 1, 1)
    grid_sel = (1, 1)
    
    def f_to_optimize(p):
        p_d = gpa.to_gpu(array([p])).astype(np.float32)
        spread_f_one_param(p_d, x_d, res_d, block=block_sel, grid=grid_sel)
        guess_d = cublas.cublasSdot_d(res_d, w_d)
        return guess_d

    return brenth(lambda p: f_to_optimize(p) - target, 0.01, 100.)


def gh_cva_quad_2d_trispread_gpu(x, w, p_1, p_2):
    """
    # evaluation of multiple spread options
    # x ... vertices of 1d Gauss-hermite abscisses
    # w ... weights of 1d Gauss-hermite abscisses
    """
    p_1_d = gpa.to_gpu(p_1).astype(np.float32)
    p_2_d = gpa.to_gpu(p_2).astype(np.float32)
    p_l = gpa.to_gpu(array([100])).astype(int)
    x_d = gpa.to_gpu(x.astype(np.float32))
    w_d = gpa.to_gpu(w).astype(np.float32)
    #ones_d = gpa.to_gpu( ones(len(x)) ).astype(np.float32)
    res_d = gpa.to_gpu(np.zeros((len(p_1), len (p_2)))).astype(np.float32)
    # computes the spread option 
    spread_file = open(config.work_dir + "spread_fun.cu").read()
    spread_module = config.SourceModule(spread_file)
    spread_f_2d = spread_module.get_function("spread_f_2d")  # spread function

    p2_l = len(p_2)
    p1_l = len(p_1)

    block_sel = (p2_l, 1, 1)
    grid_sel = (p1_l, 1)

    spread_f_2d (p_1_d, p_2_d, x_d, w_d, res_d, p_l, 
                 block=block_sel, grid=grid_sel)

    return res_d


def gh_2d_integr_cpu_wrap(arg, **kwarg):
    return gh_2d_integr_cpu(*arg, **kwarg)


# performs cpu integration
# written for multiprocessing 
def gh_2d_integr_cpu(p_x_w):
    p1, p2_v, x, w, pr_size = p_x_w 

    ones_col = np.exp(x.reshape((pr_size, 1)))
    ones_row = np.exp(x)
    w_row = w
    w_col = w.reshape((pr_size, 1))
    w_col_row = np.kron(w_col, w_row)
    p0_mat = np.kron(ones_col, np.ones(pr_size) )
    p1_mat = np.kron(np.ones(pr_size).reshape((pr_size, 1)), ones_row)
    f_o = lambda p: np.sum(w_col_row * maximum(p[0] * p0_mat + p[1] * p1_mat - 1., 0.))
    res = np.zeros(len(p2_v))
    
    for  p2_ind, p2 in enumerate(p2_v):
        res[p2_ind] = f_o(array([p1,p2]))

    return res


def gh_quad_2d_trispread_gpu(x, w, target):
    """
    # same as above, just using GPU,
    # spread option is coded directly in the
    # x ... vertices of 1d Gauss-hermite abscisses
    # w ... weights of 1d Gauss-hermite abscisses
    """
    #p_d = gpa.to_gpu (array([1.,1.])).astype(np.float32)
    x_d = gpa.to_gpu(x.astype(np.float32))
    w_d = gpa.to_gpu(w).astype(np.float32)
    ones_d = gpa.to_gpu(np.ones(len(x))).astype(np.float32)
    res_d = gpa.to_gpu(np.zeros((len(x), len (w)))).astype(np.float32)
    # computes the spread option 
    spread_file = open(config.work_dir + "spread_fun.cu").read()
    spread_module = config.SourceModule(spread_file)
    spread_f_2d = spread_module.get_function("spread_f_2d_one_param")  # spread function
    block_sel = (len(x), 1, 1)
    grid_sel = (len(w), 1)

    def f_to_optimize (p):
        p_d = gpa.to_gpu(p).astype(np.float32)
        spread_f_2d(p_d, x_d, w_d, res_d, block=block_sel, grid=grid_sel)
        guess_d = cublas.cublas_quadf(res_d, ones_d, ones_d)
        return (guess_d - target)**2
    
    res_gpu = fmin(f_to_optimize, array([1., 1.]))
    return res_gpu
