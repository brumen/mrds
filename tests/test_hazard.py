import config
from numpy import *
import mrds
import hazard
import unittest 
import pickle

# main testing class 
class hazard_tests(unittest.TestCase):

    # reads the market object 
    def setUp (self):
        print "Creating object"
        self.mo = pickle.load (open ('U:/reports/cm/reviews/extendible/contingent_extendible/python/mobj/wti_skew.obj') ) # loading the calibrated object
        
    def tearDown(self):
        print "Destroying object"
        self.mo = None

    def test_unknown_1(self):
        # cds maturity times 
        T_v = array([0.723287671,
                     1.22739726,
                     2.22739726,
                     3.22739726,
                     4.22739726,
                     5.230136986,
                     7.230136986,
                     10.23287671,
                     15.23561644])

        DF = hazard.set_DF(170)

        # cds tenors 
        cds_tenors = array([0.5, 1, 2, 3, 4, 5, 7, 10, 15])

        # cds rates 
        rates = array([0.0033, 0.0044, 0.00695, 0.0086, 0.009975, \
                       0.0114, 0.01335, 0.0152, 0.0160])
        
        hazard.solve_cds (rates, T_v, DF, {"R":0.25, "Ti_v": cds_tenors})

        self.assertTrue(True)


suite = unittest.TestLoader().loadTestsFromTestCase(hazard_tests)
unittest.TextTestRunner(verbosity=2).run(suite)
