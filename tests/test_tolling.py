#
# tests the functions regarding the tolling model
#

import datetime

from unittest import TestCase

from mrds.tolling.tolling import TollingModel


class TestTolling(TestCase):
    """ Test class for the TollingModel.
    """

    def test_smoke(self):
        """ Run smoke test on the tolling model.
        """

        tm = TollingModel.from_db( datetime.date(2015, 4, 1)
                                 , ['WTI', 'BRENT', ]
                                 , ['WTI', 'BRENT', ]
                                 , {'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,)}
                                 , { 'WEEKDAY': [('WTI', 8), ('BRENT', 16), ]
                                   , 'WEEKEND': [('WTI', 16), ('BRENT', 8), ]}
                                 , 'BRENT'
                                 , ['WTI', 'BRENT', ] )
        tm.multi_thread_calib = False  # Multi-threading doesnt allow for pickling of ORM session
        res = tm.dispatch_all(datetime.date(2015, 5, 1), datetime.date(2015, 6, 20) )

        self.assertTrue(True)

    def test_smoke_2(self):
        """ Run smoke test on the tolling model for more realistic data.
        """

        tm = TollingModel.from_db( datetime.date(2015, 4, 1)
                                 , ['ATSI-PEAK', 'ATSI_7X8', 'NG_MICHCON_GD-PEAK']
                                 , ['ATSI-PEAK', 'ATSI_7X8', 'NG_MICHCON_GD-PEAK']
                                 , {'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,)}
                                 , { 'WEEKDAY': [('ATSI-PEAK', 8), ('ATSI_7X8', 16), ]
                                   , 'WEEKEND': [('ATSI-PEAK', 16), ('ATSI_7X8', 8), ]}
                                 , 'NG_MICHCON_GD-PEAK'
                                 , ['ATSI-PEAK', 'ATSI_7X8', 'NG_MICHCON_GD-PEAK']
                                 , cuda_ind = True )
        tm.multi_thread_calib = False  # Multi-threading doesnt allow for pickling of ORM session
        import time

        t1 = time.time()
        res = tm.dispatch_all(datetime.date(2015, 5, 1), datetime.date(2015, 6, 20), nb_simulations=5000)
        print(time.time() -t1)

        self.assertTrue(True)


# running the above tests 
# suite = unittest.TestLoader().loadTestsFromTestCase(TollingSimpleTest)
# unittest.TextTestRunner(verbosity=2).run(suite)
