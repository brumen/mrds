# Compute the nearest correlation matrix,
# as defined in a paper by
# Nicholas Higham  "Computing the Nearest Correlation matrix - A Problem in Finance"
#

import numpy as np
from numpy import diag, eye, dot, sqrt, zeros
import scipy
import scipy.linalg  # linear algebra


# computes U projection
def u_proj(A, W):
    W_inv = scipy.linalg.inv(W)
    W_inv_had = W_inv * W_inv  # hadamard of W_inv
    theta = scipy.linalg.solve(W_inv_had, diag(A - eye(A.shape[0])))
    return A - dot(W_inv, dot(diag(theta), W_inv))


def s_proj(A, W):
    """
    computes the projection matrix

    """

    # positive part of A as defined
    U,S,VT = scipy.linalg.svd(W)
    W_one_half = dot(dot(U, diag(sqrt(S))), VT)
    W_inv_one_half = scipy.linalg.inv(W_one_half)
    A_eig_v, A_eig_m = scipy.linalg.eig(dot(W_one_half, dot(A, W_one_half)))
    W_positive = dot(A_eig_m, dot(diag(0.5 * (A_eig_v + np.abs(A_eig_v))), A_eig_m.transpose()))

    return dot(W_inv_one_half, dot(W_positive, W_inv_one_half))


def near_corr(A, W, nb_iter):
    """
    # iterative algorithm
    """
    S = zeros(A.shape)
    Y = A
    for ind in range(nb_iter):
        R = Y - S
        X = s_proj(R, W)
        S = X - R
        Y = u_proj(X, W)
    return Y.real  # round away the zeros


def near_corr_simple(A, nb_iter=10):
    """
    same as in the function near_corr, with W is identity matrix

    """

    return near_corr(A, eye(A.shape[0]), nb_iter)
