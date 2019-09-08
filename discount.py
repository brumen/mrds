import datetime
import numpy as np

from scipy.interpolate import splrep, splev

from ds import get_forward_curve, fwd_codes
from convert_date import d2s


def read_discount_curve(mkt_date: datetime.date, dcf=365.25):
    """ Returns the discount curve on date date_.

    :param mkt_date: market date
    :param dcf: day-count factor.
    :returns:
    """

    disc_tenors, discount_yields = get_forward_curve('DISCOUNT', mkt_date)
    diffs = [tenor - mkt_date for tenor in disc_tenors]
    disc_tenors_numeric = np.array([float(elt.days) for elt in diffs]) / dcf
    discount_yields = np.array([float(x) for x in discount_yields])

    return splrep( disc_tenors_numeric
                 , np.exp(-disc_tenors_numeric * discount_yields))  # interpolation function


def code_to_date(code_ : str) -> str:
    """ Converts fwd code (z15) into date 20151201.

    :param code_: forward code.
    """

    return ('20' + code_[1:3]) + d2s(fwd_codes[code_[0]]) + '01'


def DF_single( mktDate: datetime.date
             , fwdDate: datetime.date
             , dcf           = 365.25
             , discountCurve = None ) -> float:
    """ Discount factor for a single forward date fwd_date.

    :param mktDate: market date of the yield curve.
    :param fwdDate: forward date.
    :param dcf: date count convention
    :param discountCurve: discount curve, if provided.
    """

    return splev( (fwdDate - mktDate).days / dcf
                , discountCurve if discountCurve else read_discount_curve(mktDate) )


def DF(mkt_date: datetime.date, fwd_date):
    """ Discount factor from date date_ till date_fut.

    :param mkt_date: market date for discount factor
    :param fwd_date: forward date to which the discount is constructed.
    :returns: discount between two dates
    """

    if isinstance(fwd_date, list):
        return [DF_single(mkt_date, date_f) for date_f in fwd_date]

    return DF_single(mkt_date, fwd_date)
