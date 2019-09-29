
import datetime
import numpy as np

from typing import List, Dict, Tuple

import ds
from forward_curve import FwdCurve
from vols.vols import Volatility


class C0C1C2Volatility(Volatility):
    """ c0-c1-c2 volatility parametrization
        _smooth_ind is the smoothness indicator
        _alpha is the smoothness factor
    """

    def __init__(self
                 , com_name : str
                 , mkt_date : datetime.date
                 , fwd_params : FwdCurve
                 , vol_params : Dict[datetime.date, List]):
        super(C0C1C2Volatility, self).__init__(com_name, mkt_date, fwd_params=fwd_params, vol_params=vol_params)
        self.__vol_dates = list(vol_params.keys())  # TODO: CHECK THIS PART

    @classmethod
    def from_db(cls, com_name : str, mkt_date : datetime.date):
        """ Reads the forward and vol curve from external source.

        :param com_name: name of the commodity one wants, e.g. 'WTI', ...
        :param mkt_date: for which market date the vol is needed
        """

        fwd_curve = FwdCurve.from_db(mkt_date, com_name)

        # TODO: FAKING THE CURVE HERE A BIT
        com_dates, _ = ds.get_forward_curve(com_name, mkt_date)
        com_vol = {date_: (0.1, 0.2, 0.3, 0.4, 0.5)  # c0, c1, c2, theta, alpha
                    for date_ in com_dates }

        return cls( com_name
                  , mkt_date
                  , fwd_params = fwd_curve
                  , vol_params = com_vol )

    @property
    def _vol_dates(self) -> List[datetime.date]:
        return self.__vol_dates

    def _get_next_date(self, fwd_date : datetime.date ) -> datetime.date:
        """ Returns the next date on the forward curve after fwd_date.

        :param fwd_date: the date after which we are searching on the curve.
        :returns: date after the fwd_date on the forward curve
        """

        fd_better = [fd for fd in self._vol_dates if fd > fwd_date]

        if fd_better:  # the list of larger dates is not empty
            return fd_better[0]

        # else returns the last date on the curve
        return self._vol_dates[-1]

    def _get_c0c1c2(self, ttm : datetime.date) -> Tuple[float, float, float, float, float]:
        next_date = self._get_next_date(ttm)
        return self._vol_params[next_date]

    def implied_vol(self, fwd_value : float, ttm : datetime.date, smooth_ind=True) -> float:
        """ Computes the implied volatility of quadratic volatility surface.

        :param fwd_value: forward value for which to compute
        :param ttm: time-to-maturity
        :param smooth_ind: indicator whether to smooth the curve.
        """

        atm_strike = self._fwd_params.fwd_value(ttm)
        z = np.log(fwd_value / atm_strike)

        c0, c1, c2, theta, alpha = self._get_c0c1c2(ttm)

        v = c0 + c1 * z + c2**2 * z**2
        sigma_star = c0 * theta - alpha * (c0 * theta - c0)
        a = c0 * theta - sigma_star

        # TODO: CHECK IF BELOW IS arctan or arctan2
        if smooth_ind:
            return v if v < sigma_star else 2. * a / np.pi * np.arctan ( np.pi / (2 * a) * (v - sigma_star) ) + sigma_star

        # smooth_ind == False, some additional logic
        if v >= c0 * theta:
            return c0 * theta

        return v
