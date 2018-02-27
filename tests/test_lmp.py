import config
import numpy as np
import lmp

gl_simple = [(1, 150., 0., 10.), (2, 150., 120., 15.)]
pl_simple = [(1, 2, 100.)]
# res = lmp.comp_val(gl_simple, pl_simple, show_sol=True)
# print res
# print lmp.comp_lmp(gl_simple, pl_simple, show_sol=False)

gl_1 = [(1, 500., 0., 10.), (2, 0., 75., 0.), (3, 500., 325, 20.)]
pl_1 = [(1, 2, 300.), (2, 3, 40.), (1, 3, 200.)]
# print lmp.comp_val(gl_1, pl_1, debug_ind=False)
# print lmp.comp_lmp(gl_1, pl_1, show_sol=False)


# complicated network # (node, gen, load, price)
gl_2 = [(1, 300., 100., 10.), (2, 0., 100., 0.), (3, 0., 100., 0.),
        (4, 300., 100., 20.), (5, 0., 100., 0.)]
pl_2 = [(1, 2, 500.), (1, 3, 200.), (2, 3, 200.), (3, 4, 500.), (4, 5, 150.)]
print lmp.comp_val(gl_2, pl_2)
print lmp.comp_lmp(gl_2, pl_2, show_sol=True)


# large network
N = 20
gl_3 = zip(range(1, N+1), np.random.random(N) * 100.,
           np.random.random(N) * 50.,
           np.random.random(N) * 100.)  # (n, gen, load, p) , (2, 0., 100., 0.), (3, 0., 100., 0.),
pl_3 = []
for i in range(1, N+1):
    for j in range(i+1, N+1):
        pl_3.append((i, j, 2.))
# print lmp.comp_lmp(gl_3, pl_3, show_sol=True, solver='glpk')

# print lmp.find_pos_y(2,1, gl_2, pl_2)
# print lmp.find_connected(5, pl_2)
