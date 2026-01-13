import datetime
import numpy as np

from mrds.errors import ComSkewError


class MrdsDiscount:

    def __init__(self, mkt_date: datetime.date, dc, dcf=252.0):
        self._discount_curve = dc
        self.dcf = dcf
        self.mkt_date = mkt_date

    def _difference_to_market_date(self, fwd_date: datetime.date) -> float:
        """Computes the difference to market date.

        :param fwd_date: date to compute the distance to market date.
        """

        return (fwd_date - self.mkt_date).days / self.dcf

    def DF(self, fwd_time: [float, datetime.date]):
        """Discount from self.mkt_date to fwd_time. Using basic discount curve.

        :param fwd_time: future time to discount to. can be '20140101', ...
        """

        if (type(fwd_time) is np.double) or (type(fwd_time) is float):
            time_diff = fwd_time

        elif isinstance(fwd_time, datetime.date):
            time_diff = self._difference_to_market_date(fwd_time)

        else:
            raise ComSkewError(
                "fwd_time given in function DF is not of form [float, datetime.date]"
            )

        return self._discount_curve(time_diff)
