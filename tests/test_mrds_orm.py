# test cases for the base mrds module.
# IMPORTANT: date has to be _before_ 2015-04-20, CHOOSE 2015-04-01 by default

import datetime
from unittest import TestCase

from mrds.mrds_orm import ComSkewORM

MKT_DATE = datetime.date(2015, 4, 1)


class TestComSkewORMMultiple(TestCase):
    """ Test for Com skew model for multiple assets.
    """

    MKT_DATE = MKT_DATE

    def test_calibrate_multiple(self):
        """ Tests whether the model even calibrates to multiple assets.
        """

        model = ComSkewORM.from_db(self.MKT_DATE, ('WTI', 'BRENT', ))
        # model.multi_thread_calib = False
        res1 = model.simulate_curves( ('WTI', 'BRENT', )
                                    , 1000
                                    , [datetime.date(2015, 5, 1), datetime.date(2015, 6, 1), ]  # simulation times
                                    , [datetime.date(2015, 5, 20), datetime.date(2015, 7, 15), ]  # tenor idx to simulate
                                    , )

        self.assertTrue(True)
