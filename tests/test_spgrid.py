## First lines are for Python in Morgan Stanley
## see also http://wiki.ms.com/twiki/cgi-bin/view/Python/InstalledModules
import config 
from numpy import *
import scipy
import scipy.integrate
import scipy.linalg
import matplotlib
import unittest
import pickle

# testing sparse grids
import sg


class sg_tests(unittest.TestCase):

    def test_unknow_1(self):
        A = sg.sg_p (1,5)
        B = sg.sg_w (1,5)
        f = lambda x: sum (x**2)
        res1 = sg.sg_quad (1, 15, f)
        res2 = scipy.integrate.quad (f, 0., 1.)[0]
        res3 = scipy.linalg.norm(res1-res2) < 1.

        # really WEAK test
        self.assertTrue( res3 )


suite = unittest.TestLoader().loadTestsFromTestCase(sg_tests)
unittest.TextTestRunner(verbosity=2).run(suite)
