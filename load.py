#
# Load generation network
#

import numpy             as np
import networkx          as nx
import matplotlib.pyplot as plt

from scipy.optimize import linprog
from typing         import List, Tuple, Set


class Load:
    """ Computes the load distribution on the given network.
    """

    def __init__(self
                 , generation_nodes : List[Tuple[int, float, float]]
                 , network_struct   : List[Tuple[int, int, float, float]]):
        """

        :param generation_nodes: graph in the form of: (node, generation or load, price). If generation is positive,
                   it's generation, otherwise it's load. Price is always positive.
        :param network_struct: structure of the network (first_node, second_node, capacity on that bus, cost of transmission)
        """

        self.__generation_nodes = generation_nodes
        self.__network_struct   = network_struct

        # cached stuff
        self.__all_nodes = None  # all nodes extracted from the network

    @property
    def _all_nodes(self) -> Set:
        """ Extract the nodes from the network.
        """

        if self.__all_nodes:
            return self.__all_nodes

        self.__all_nodes = set([])

        # go through generation_nodes list
        for node, _, _ in self.__generation_nodes:
            self.__all_nodes.add(node)

        for node_1, node_2, _, _ in self.__network_struct:
            self.__all_nodes.add(node_1)
            self.__all_nodes.add(node_2)

        return self.__all_nodes

    def __find_node(self, node):
        """ Finds the node node in the graph generation_nodes.

        :param node: name of the node you are searching in graph generation_nodes
        """

        pot_res_raw = [(gen_load, price) for (node_1, gen_load, price) in self.__generation_nodes
                       if node is node_1]

        return None if not pot_res_raw else pot_res_raw[0]

    def loads(self):
        """ Constructs the linear program to determine the load distribution.
        """

        nb_vars = 2 * len(self.__network_struct)
        gen_load_nodes = [node for (node, gen_load, price) in self.__generation_nodes]

        A_ineq = []
        b_ineq = []
        A_eq   = []
        b_eq   = []

        for node in self._all_nodes:
            if node in gen_load_nodes:  # some gen., load on these nodes
                gen_load, price = self.__find_node(node)
                row_gen_load = np.zeros(nb_vars)
                row_gen_load2 = np.zeros(nb_vars)

                for nb_conn, (orig, dst, cap, trans) in enumerate(self.__network_struct):
                    nb_conn_2 = 2 * nb_conn
                    if orig is node:
                        if gen_load >= 0.:  # generation
                            row_gen_load[nb_conn_2] = 1.
                            row_gen_load2[nb_conn_2 + 1] = 1.
                        else:
                            row_gen_load[nb_conn_2+1] = 1.
                            row_gen_load2[nb_conn_2] = 1.
                    if dst is node:
                        if gen_load >= 0.:
                            row_gen_load[nb_conn_2+1] = 1.
                            row_gen_load2[nb_conn_2] = 1.
                        else:
                            row_gen_load[nb_conn_2] = 1.
                            row_gen_load2[nb_conn_2+1] = 1.

                A_ineq.append(row_gen_load)
                b_ineq.append(np.abs(gen_load))
                A_eq.append(row_gen_load2)
                b_eq.append(0.)
            else:  # transmission nodes
                row_transm = np.zeros(nb_vars)
                for nb_conn, (orig, dst, cap, trans) in enumerate(self.__network_struct):
                    if orig is node:
                        row_transm[(2*nb_conn):(2*nb_conn+2)] = (1., -1.)
                    if dst is node:
                        row_transm[(2*nb_conn):(2*nb_conn+2)] = (-1., 1.)
                A_eq.append(row_transm)
                b_eq.append(0.)

        opt_vec = np.zeros(nb_vars)
        for nb_conn, (orig, dst, cap, trans) in enumerate(self.__network_struct):
            nb_conn_2 = 2 * nb_conn
            lr_row = np.zeros(nb_vars)
            lr_row[nb_conn_2:(nb_conn_2 + 2)] = (1, 1)
            # TODO: check if this can be rewritten w/ equalities.
            A_ineq.append(lr_row)
            b_ineq.append(cap)
            A_ineq.append(-lr_row)
            b_ineq.append(cap)
            opt_vec[nb_conn_2:(nb_conn_2+2)] = (-trans, -trans)
            orig_load_price = self.__find_node(orig)
            dest_load_price = self.__find_node(dst)

            if orig_load_price:
                orig_load, orig_price = orig_load_price
                if orig_load >= 0.:  # generation
                    opt_vec[nb_conn_2] -= orig_price
                else:
                    opt_vec[nb_conn_2+1] += orig_price

            if dest_load_price:
                dst_load, dest_price = dest_load_price
                if dst_load <= 0.:
                    opt_vec[nb_conn_2] += dest_price
                else:
                    opt_vec[nb_conn_2+1] -= dest_price

        problem = linprog( -opt_vec
                         , A_eq = np.array(A_eq)
                         , A_ub = np.array(A_ineq)
                         , b_ub = np.array(b_ineq)
                         , b_eq = np.array(b_eq) )
        solution_raw = problem.x

        solution_pres = [t_left * (t_left >= t_right) - t_right * (t_right > t_left)
                         for (t_left, t_right) in solution_raw.reshape((len(solution_raw)//2, 2))]

        return { 'value'         : problem.fun
               , 'solution'      : solution_raw
               , 'solution_edges': zip(self.__network_struct, solution_pres)}

    @staticmethod
    def draw_network(sol_edges : List[Tuple], pos = None, cutoff_value=1.):
        """ Draws the transmission network graph.

        :param sol_edges: edges of the graph to display in the form
                          (node1, node2, capacity, capacity_cost)
        :param pos:
        :param cutoff_value: the value below which the edge is not shown
        """

        nodes = set([n1 for (n1, n2, cap, cap_cost), p_sol in sol_edges] +
                    [n2 for (n1, n2, cap, cap_cost), p_sol in sol_edges])
        edges = [(n1, n2) for (n1, n2, cap, cap_cost), p_sol in sol_edges
                 if np.abs(p_sol) > cutoff_value]
        labels = ['{:.2f}'.format(p_sol) for _, p_sol in sol_edges if np.abs(p_sol) > cutoff_value]

        g = nx.Graph()
        for node in nodes:
            g.add_node(node)

        for edge_start, edge_end in edges:
            g.add_edge(edge_start, edge_end)

        pos_used = nx.shell_layout(g) if pos is None else pos

        nx.draw(g, pos_used)
        nx.draw_networkx_edge_labels(g, pos_used, edge_labels=dict(zip(g.edges(), labels)))
        plt.show()
