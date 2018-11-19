#
# computes location marginal pricing (LMP)
#

import numpy as np
from scipy.optimize import linprog

import copy


# nodes in the network are buses
# variables
def find_pos_y(node_1, node_2, gl, pl):
    """
    find the position of (node_1, node_2) connection in PL
    :param pl: list of (node_1, node_2, c_12)
    """

    n1, n2, c = pl[0]
    k = 0
    while ((node_1, node_2) != (n1, n2)) and ((node_2, node_1) != (n1, n2)):
        k += 1
        n1, n2, c = pl[k]

    return len(gl) + k


def find_incoming(node_nb, pl):
    """
    find incoming nodes into node_nb
    """
    res_l = []
    for node_1, node_2, c_12 in pl:
        if node_2 == node_nb:
            res_l.append(node_1)
    return res_l


def find_outgoing(node_nb, pl):
    """
    find incoming nodes into node_nb
    """
    res_l = []
    for node_1, node_2, c_12 in pl:
        if node_1 == node_nb:
            res_l.append(node_2)
    return res_l


def find_connected(node_nb, pl):
    res_l = []
    for node_1, node_2, c_12 in pl:
        if node_1 == node_nb:
            res_l.append(node_2)
        if node_2 == node_nb:
            res_l.append(node_1)
    return res_l


def comp_val(gl, pl, show_sol=False):
    """
    computes the value of the optimization problem given gl and pl
    :param show_sol: shows the solution
    """

    nb_nodes = len(gl)
    nb_lines = len(pl) # each line counted 2x
    nb_vars = nb_nodes + nb_lines

    A_ineq = []
    b_ineq = []
    A_eq = []
    b_eq = []

    # lower/upper bound on x, transmission
    lb = np.empty(nb_vars)  # lower bound
    lb[:nb_nodes] = 0.
    ub = np.empty(nb_vars)
    ub[:nb_nodes] = np.array([gen for (node, gen, load, p) in gl])  # generation upper boundary
    # line constraints ( abs(y) < c )
    for node_1, node_2, c in pl:
        y_pos = find_pos_y(node_1, node_2, gl, pl)
        lb[y_pos] = -c
        ub[y_pos] = c

    # node constraints
    for node_nb, generation, load, gen_price in gl:
        vt_n = np.zeros(nb_vars)
        incoming_l = find_incoming(node_nb, pl)
        outgoing_l = find_outgoing(node_nb, pl)
        vt_n[node_nb-1] = -1.
        for node_inc in incoming_l:
            vt_n[find_pos_y(node_inc, node_nb, gl, pl)] = -1.
        for node_out in outgoing_l:
            vt_n[find_pos_y(node_nb, node_out, gl, pl)] = 1.
        A_ineq.append(vt_n)
        b_ineq.append(-load)

    # optimization vector - prices over nodes
    opt_vec = np.zeros(nb_vars)
    opt_vec[:nb_nodes] = np.array([p for (node_nb, gen, load, p) in gl])

    if A_eq:  # linprog cannon handle empty matrices
        problem = linprog(opt_vec, A_ub=np.array(A_ineq), b_ub=np.array(b_ineq), A_eq=A_eq, b_eq=b_eq
                         , bounds = list(zip(lb, ub)))
    else:
        problem = linprog(opt_vec, A_ub=np.array(A_ineq), b_ub=np.array(b_ineq)
                          , bounds=list(zip(lb, ub)))

    sol_v = problem.x
    if show_sol:
        # bus generation: first nb_nodes
        print("Bus: ", sol_v[:nb_nodes])
        for line_idx, (node_1, node_2, c) in enumerate(pl):
            print("Line", (node_1, node_2), "transm:", sol_v[nb_nodes + line_idx])

    return problem.fun


def comp_lmp(gl, pl, show_sol=False, debug_ind=False, solver='glpk'):
    """
    Compute locational marginal pricing.

    """
    comp_basic = comp_val(gl, pl, show_sol=show_sol)
    lmp = np.empty(len(gl))
    for node_nb, generation, load, gen_price in gl:
        # glt = copy.deepcopy(gl)
        gl_orig = copy.deepcopy(gl[node_nb-1])
        gl[node_nb-1] = (node_nb, generation, load + 1., gen_price)
        new_value = comp_val(gl, pl, show_sol=show_sol)
        gl[node_nb-1] = gl_orig
        lmp[node_nb-1] = new_value - comp_basic

    return lmp
