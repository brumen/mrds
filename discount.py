#
#  Functions concerned w/ discount curve
#

import datetime
import numpy    as np
import QuantLib as ql

from scipy.interpolate import splrep, splev

from ds import get_forward_curve


class DiscountCurve:

    @staticmethod
    def discount_function(mkt_date : datetime.date, dcf = 365.25):
        """ Returns the discount function for the market date.

        :param mkt_date: market date
        :param dcf: day-count factor.
        :returns:
        """

        disc_tenors, discount_yields = get_forward_curve('DISCOUNT', mkt_date)
        diffs = [tenor - mkt_date for tenor in disc_tenors]
        disc_tenors_numeric = np.array([float(elt.days) for elt in diffs]) / dcf
        discount_yields = np.array([float(x) for x in discount_yields])

        interpolated_curve = splrep(disc_tenors_numeric, np.exp(-disc_tenors_numeric * discount_yields))  # interpolation function

        return lambda time_diff: splev(time_diff, interpolated_curve)

    @staticmethod
    def discount_function_ql(mkt_date : datetime.date):
        """ Returns the Quantlib discount function for the market date.

        :param mkt_date: market date
        :param dcf: day-count factor.
        :returns:
        """

        eonia_rates = get_forward_curve('DISCOUNT_QL', mkt_date)
        ql.Settings.instance().evaluationDate = ql.Date(mkt_date)
        eonia_helpers = [ql.OISRateHelper(2, ql.Period(*tenor), ql.QuoteHandle(ql.SimpleQuote(rate / 100)), ql.Eonia())
                         for rate, tenor in eonia_rates]

        eonia_curve_flat = ql.PiecewiseFlatForward(0, ql.TARGET(), eonia_helpers, ql.Actual365Fixed())
        eonia_curve_flat.enableExtrapolation()

        return lambda time_diff: eonia_curve_flat.discount(time_diff)
