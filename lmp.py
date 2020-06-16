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

    def __init__( self
                , generation : List[Tuple[str, float, float, float]]
                , network    : List[Tuple[str, str, float]]):
        """ Nodes in the network are buses.

        :param generation: generation list in the form (node, generation, load, price of generation)
        :param network: in the form of (node_1, node_2, transmission capacity)
        """

        self.generation = generation  # generation list gl
        self.network    = network  # network list pl

        self.__all_nodes_int = None

    @property
    def __all_nodes(self) -> List:
        """ Constructs all the nodes in the network.

        :returns: set of all nodes
        """

        if self.__all_nodes_int:
            return self.__all_nodes_int

        # generate all nodes
        self.__all_nodes_int = set()
        for node, _, _, _ in self.generation:
            self.__all_nodes_int.add(node)
        for node_1, node_2, _ in self.network:
            self.__all_nodes_int.add(node_1)
            self.__all_nodes_int.add(node_2)

        return list(self.__all_nodes_int)

    def __nodes_to_integer(self, node : str) -> int:
        """ Returns the integer index of the node

        :param node: node for which index is searched.
        """

        return self.__all_nodes.index(node)

    def __integers_to_nodes(self, node_int : int) -> str:
        """ Returns the node name from its integer number.

        :param node_int: integer number of the node.
        """

        return self.__all_nodes[node_int]

    def __find_pos_y(self, node_1 : str, node_2 : str) -> int:
        """ Find the position of (node_1, node_2) connection in generation list generation.

        :param node_1: origin node that we are searching in self.network
        :param node_2: destination node we are searching the connection from.
        """

        n1, n2, _ = self.network[0]
        k = 0
        while ((node_1, node_2) != (n1, n2)) and ((node_2, node_1) != (n1, n2)):
            k += 1
            n1, n2, _ = self.network[k]

        return len(self.generation) + k

    def __find_incoming(self, node_nb : str) -> List[str]:
        """ Find incoming nodes into node_nb in generation list self.generation

        :param node_nb: node to search.
        """

        return [node_1 for node_1, node_2, _ in self.network if node_2 == node_nb ]

    def __find_outgoing(self, node_nb : str) -> List[str]:
        """ Find incoming nodes into node_nb.

        :param node_nb: node to search.
        :returns: list of incoming nodes into node_nb.
        """

        return [node_2 for node_1, node_2, _ in self.network
                if node_1 == node_nb]

    def __find_connected(self, node_nb : str) -> List[str]:
        """ Find all nodes connected to node_nb, either incoming or outgoing.

        :param node_nb: node for which the connections are searched.
        """

        res_l = []
        for node_1, node_2, _ in self.network:
            if node_1 == node_nb:
                res_l.append(node_2)
            if node_2 == node_nb:
                res_l.append(node_1)

        return res_l

    def compute_load_distribution(self):
        """ Computes the value of the optimization problem given generation_nodes and pl.
        """

        nb_nodes = len(self.generation)
        nb_lines = len(self.network)  # each line counted 2x
        nb_vars = nb_nodes + nb_lines

        A_ineq = []
        b_ineq = []
        A_eq   = []
        b_eq   = []

        # lower/upper bound on x, transmission
        lb = np.empty(nb_vars)  # lower bound
        lb[:nb_nodes] = 0.
        ub = np.empty(nb_vars)
        ub[:nb_nodes] = np.array([gen for (node, gen, load, p) in self.generation])  # generation upper boundary
        # line constraints ( abs(y) < c )
        for node_1, node_2, c in self.network:
            y_pos = self.__find_pos_y(node_1, node_2)
            lb[y_pos] = -c
            ub[y_pos] = c

        # node constraints
        for node_nb, generation, load, gen_price in self.generation:
            vt_n = np.zeros(nb_vars)
            incoming_l = self.__find_incoming(node_nb)
            outgoing_l = self.__find_outgoing(node_nb)
            vt_n[self.__nodes_to_integer(node_nb)-1] = -1.
            for node_inc in incoming_l:
                vt_n[self.__find_pos_y(node_inc, node_nb)] = -1.
            for node_out in outgoing_l:
                vt_n[self.__find_pos_y(node_nb, node_out)] = 1.
            A_ineq.append(vt_n)
            b_ineq.append(-load)

        # optimization vector - prices over nodes
        opt_vec = np.zeros(nb_vars)
        opt_vec[:nb_nodes] = np.array([p for (node_nb, gen, load, p) in self.generation])

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
        nb_nodes = len(self.generation)
        sol_v = problem.x

        # bus generation: first nb_nodes
        print('Bus: {0}'.format(sol_v[:nb_nodes]))
        for line_idx, (node_1, node_2, c) in enumerate(self.network):
            print('Line {0}, {1}, transm: {2}'.format(node_1, node_2, sol_v[nb_nodes + line_idx]))

    def compute_lmp(self, show_sol=False):
        """ Compute locational marginal pricing.

        """

        comp_basic = self.compute_load_distribution().fun  # value of the optimization function
        lmp = np.empty(len(self.generation))

        for node_nb, generation, load, gen_price in self.generation:
            node_int = self.__nodes_to_integer(node_nb)
            gl_orig = copy.deepcopy(self.generation[node_int-1])
            self.generation[node_int-1] = (node_nb, generation, load + 1., gen_price)
            new_value = self.compute_load_distribution().fun
            self.generation[node_int-1] = gl_orig
            lmp[node_int-1] = new_value - comp_basic

        return lmp
