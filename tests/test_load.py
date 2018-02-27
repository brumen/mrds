import config
import numpy as np
import load

gl_compl = [(1, 100, 30), (2, 100, 35),
            (5, -150, 70), (6, -150, 50)]
p_compl = [(1, 3, 100, 1),
           (2, 4, 100, 1),
           (3, 4, 200, 1),
           (3, 5, 120, 1),
           (4, 6, 200, 1)]

gl_compl_pos = {1: np.array([0., 0.]),
                2: np.array([0., -1.]),
                3: np.array([1., 0.]),
                4: np.array([1., -1.]),
                5: np.array([2., 0.]),
                6: np.array([2., -1.])}

res_compl = load.cons_lp(6, gl_compl, p_compl)
load.draw_network(6, gl_compl, res_compl['solution_edges'], pos=gl_compl_pos)
