#
# computes location marginal pricing (LMP)
#

import copy
import numpy as np
from scipy.optimize import linprog
from typing         import List, Tuple


class LMP:
    """ Locational marginal pricing.
    """

    def __init__(self, gl : List[Tuple], pl : List[Tuple]):
        """
        nodes in the network are buses
        variables

        :param gl: generation list in the form (node, generation, load, price of generation)
        :param pl: in the form of (node_1, node_2, transmission capacity)
        """

        self.gl = gl
        self.pl = pl

    def __find_pos_y(self, node_1, node_2):
        """ Find the position of (node_1, node_2) connection in production list pl.

        :param pl: list of (node_1, node_2, c_12)
        """

        n1, n2, c = self.pl[0]
        k = 0
        while ((node_1, node_2) != (n1, n2)) and ((node_2, node_1) != (n1, n2)):
            k += 1
            n1, n2, c = self.pl[k]

        return len(self.gl) + k

    def __find_incoming(self, node_nb):
        """ Find incoming nodes into node_nb in production list pl

        :param node_nb: node to search.
        """

        res_l = []
        for node_1, node_2, c_12 in self.pl:
            if node_2 == node_nb:
                res_l.append(node_1)

        return res_l

    def __find_outgoing(self, node_nb):
        """ Find incoming nodes into node_nb
        """

        res_l = []
        for node_1, node_2, c_12 in self.pl:
            if node_1 == node_nb:
                res_l.append(node_2)
        return res_l

    def __find_connected(self, node_nb):
        res_l = []
        for node_1, node_2, c_12 in self.pl:
            if node_1 == node_nb:
                res_l.append(node_2)
            if node_2 == node_nb:
                res_l.append(node_1)

        return res_l

    def compute_load_distribution(self):
        """ Computes the value of the optimization problem given generation_nodes and pl.
        """

        nb_nodes = len(self.gl)
        nb_lines = len(self.pl)  # each line counted 2x
        nb_vars = nb_nodes + nb_lines

        A_ineq = []
        b_ineq = []
        A_eq = []
        b_eq = []

        # lower/upper bound on x, transmission
        lb = np.empty(nb_vars)  # lower bound
        lb[:nb_nodes] = 0.
        ub = np.empty(nb_vars)
        ub[:nb_nodes] = np.array([gen for (node, gen, load, p) in self.gl])  # generation upper boundary
        # line constraints ( abs(y) < c )
        for node_1, node_2, c in self.pl:
            y_pos = self.__find_pos_y(node_1, node_2)
            lb[y_pos] = -c
            ub[y_pos] = c

        # node constraints
        for node_nb, generation, load, gen_price in self.gl:
            vt_n = np.zeros(nb_vars)
            incoming_l = self.__find_incoming(node_nb)
            outgoing_l = self.__find_outgoing(node_nb)
            vt_n[node_nb-1] = -1.
            for node_inc in incoming_l:
                vt_n[self.__find_pos_y(node_inc, node_nb)] = -1.
            for node_out in outgoing_l:
                vt_n[self.__find_pos_y(node_nb, node_out)] = 1.
            A_ineq.append(vt_n)
            b_ineq.append(-load)

        # optimization vector - prices over nodes
        opt_vec = np.zeros(nb_vars)
        opt_vec[:nb_nodes] = np.array([p for (node_nb, gen, load, p) in self.gl])

        if A_eq:  # linprog cannon handle empty matrices
            return linprog( opt_vec
                          , A_ub = np.array(A_ineq)
                          , b_ub = np.array(b_ineq)
                          , A_eq = A_eq
                          , b_eq = b_eq
                          , bounds = list(zip(lb, ub)) )

        return linprog( opt_vec
                      , A_ub = np.array(A_ineq)
                      , b_ub = np.array(b_ineq)
                      , bounds=list(zip(lb, ub)) )

    def show_lmp(self):
        """ Prints out the result of optimization.
        """

        problem = self.compute_load_distribution()
        nb_nodes = len(self.gl)
        sol_v = problem.x

        # bus generation: first nb_nodes
        print('Bus: {0}'.format(sol_v[:nb_nodes]))
        for line_idx, (node_1, node_2, c) in enumerate(self.pl):
            print('Line {0}, {1}, transm: {2}'.format(node_1, node_2, sol_v[nb_nodes + line_idx]))

    def compute_lmp(self, show_sol=False):
        """ Compute locational marginal pricing.

        """

        comp_basic = self.compute_load_distribution().fun  # value of the optimization function
        lmp = np.empty(len(self.gl))

        for node_nb, generation, load, gen_price in self.gl:
            gl_orig = copy.deepcopy(self.gl[node_nb-1])
            self.gl[node_nb-1] = (node_nb, generation, load + 1., gen_price)
            new_value = self.compute_load_distribution().fun
            self.gl[node_nb-1] = gl_orig
            lmp[node_nb-1] = new_value - comp_basic

        return lmp
