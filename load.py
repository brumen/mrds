import numpy as np
from scipy.optimize import linprog
import networkx as nx
import matplotlib.pyplot as plt


def find_node(node, gl):
    pot_res_raw = [(gen_load, price) for (node_1, gen_load, price) in gl
                   if node is node_1]
    if pot_res_raw == []:  # nothing found
        return None, None
    else:
        return pot_res_raw[0]


def cons_lp(nb_nodes, gl, p):

    nb_vars = 2 * len(p)
    gen_load_nodes = [node for (node, gen_load, price) in gl]
    A_ineq = []
    b_ineq = []
    A_eq = []
    b_eq = []

    for node in range(1, nb_nodes+1):
        if node in gen_load_nodes:  # some gen., load on these nodes
            gen_load, price = find_node(node, gl)
            row_gen_load = np.zeros(nb_vars)
            row_gen_load2 = np.zeros(nb_vars)
            for nb_conn, (orig, dst, cap, trans) in enumerate(p):
                if orig is node:
                    if gen_load >= 0.:  # generation
                        row_gen_load[2*nb_conn] = 1.
                        row_gen_load2[2*nb_conn + 1] = 1.
                    else:
                        row_gen_load[2*nb_conn+1] = 1.
                        row_gen_load2[2*nb_conn] = 1.
                if dst is node:
                    if gen_load >= 0.:
                        row_gen_load[2*nb_conn+1] = 1.
                        row_gen_load2[2*nb_conn] = 1.
                    else:
                        row_gen_load[2*nb_conn] = 1.
                        row_gen_load2[2*nb_conn+1] = 1.
            A_ineq.append(row_gen_load)
            b_ineq.append(np.abs(gen_load))
            A_eq.append(row_gen_load2)
            b_eq.append(0.)
        else:  # transmission nodes
            row_transm = np.zeros(nb_vars)
            for nb_conn, (orig, dst, cap, trans) in enumerate(p):
                if orig is node:
                    row_transm[(2*nb_conn):(2*nb_conn+2)] = (1., -1.)
                if dst is node:
                    row_transm[(2*nb_conn):(2*nb_conn+2)] = (-1., 1.)
            A_eq.append(row_transm)
            b_eq.append(0.)

    opt_vec = np.zeros(nb_vars)
    for nb_conn, (orig, dst, cap, trans) in enumerate(p):
        lr_row = np.zeros(nb_vars)
        lr_row[(2*nb_conn):(2*nb_conn + 2)] = (1, 1)
        A_ineq.append(lr_row)
        b_ineq.append(cap)
        A_ineq.append(-lr_row)
        b_ineq.append(cap)
        opt_vec[(2*nb_conn):(2*nb_conn+2)] = (-trans, -trans)
        orig_load, orig_price = find_node(orig, gl)
        dst_load, dst_price = find_node(dst, gl)

        if orig_load is not None:
            if orig_load >= 0.:  # generation
                opt_vec[2*nb_conn] -= orig_price
            else:
                opt_vec[2*nb_conn+1] += orig_price
        if dst_load is not None:
            if dst_load <= 0.:
                opt_vec[2*nb_conn] += dst_price
            else:
                opt_vec[2*nb_conn+1] -= dst_price

    A_eq = np.array(A_eq)
    A_ineq = np.array(A_ineq)
    b_eq = np.array(b_eq)
    b_ineq = np.array(b_ineq)

    problem = linprog(-opt_vec, A_eq=A_eq, A_ub=A_ineq, b_ub=b_ineq, b_eq=b_eq)
    solution_raw = problem.x
    solution_val = problem.fun

    solution_per_node = solution_raw.reshape((len(solution_raw)//2, 2))
    solution_pres = [t_left * (t_left >= t_right) - t_right * (t_right > t_left)
                     for (t_left, t_right) in solution_per_node]
    solution_edges = zip(p, solution_pres)

    return {'value': solution_val,
            'solution': solution_raw,
            'solution_edges': solution_edges}


def draw_network(sol_edges, pos=None):
    """
    draws the network graph

    :param sol_edges: edges of the graph solution.

    """

    nodes = set([n1 for (n1, n2, cap, cap_cost), p_sol in sol_edges] +
                [n2 for (n1, n2, cap, cap_cost), p_sol in sol_edges])
    edges = [(n1, n2) for (n1, n2, cap, cap_cost), p_sol in sol_edges
             if np.abs(p_sol) > 1.]
    labels = ["%.2f" % p_sol for (n1, n2, cap, cap_cost), p_sol in sol_edges
              if np.abs(p_sol) > 1.]

    g = nx.Graph()
    for node in nodes:
        g.add_node(node)
    for edge_start, edge_end in edges:
        g.add_edge(edge_start, edge_end)

    edge_labels = dict(zip(g.edges(), labels))
    if pos is None:
        pos_used = nx.shell_layout(g)
    else:
        pos_used = pos

    nx.draw(g, pos_used)
    nx.draw_networkx_edge_labels(g, pos_used, edge_labels=edge_labels)
    plt.show()
