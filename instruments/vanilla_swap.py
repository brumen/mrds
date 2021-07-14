""" Prices simples EU vanilla swap.
"""
import datetime
import QuantLib as ql

from typing    import List
from functools import lru_cache

from mrds.forward_curve import FwdCurve
from mrds.discount      import DiscountCurve


class VanillaSwap:
    """ Basic Vanilla Swap instrument """

    def __init__( self
                , start_date : datetime.date
                , market_date : datetime.date
                , pricing_date : datetime.date = None
                , maturity : str = '5Y'
                , frequency : str = '3M'
                , swap_rate : float = 50.
                , index_name : str = 'WTI'
                , ):

        self.start_date = start_date
        self.market_date = market_date
        self.pricing_date = pricing_date
        self.maturity = maturity
        self.frequency = frequency
        self.swap_rate = swap_rate
        self.index_name = index_name

        # cached items
        self._index_curve = None

    def payments( self ) -> List[datetime.date]:
        """ Returns the schedule of the vanilla swap.
        """

        pricing_used = self.market_date if self.pricing_date is None else self.pricing_date

        calendar = ql.TARGET()
        start    = ql.Date.from_date(self.start_date)
        maturity = calendar.advance(start, ql.Period(self.maturity))

        return [date.to_date()
                for date in ql.MakeSchedule(start, maturity, ql.Period(self.frequency))
                if ( date.to_date() >= pricing_used) and (date.to_date() != self.market_date)]  # TODO: THIS SHOULD BE NICER

    @property
    def index_curve(self):

        if self._index_curve is not None:
            return self._index_curve

        self._index_curve = FwdCurve.from_db(self.market_date, self.index_name)
        return self._index_curve

    @index_curve.setter
    def index_curve(self, new_curve):
        self._index_curve = new_curve

    def PV(self):
        """ Computes the PV.
        """

        discount_curve = DiscountCurve(self.market_date).discount_function_2

        return sum([ (self.index_curve.fwd_value(payment_date) - self.swap_rate) * discount_curve(payment_date)
                     for payment_date in self.payments() ])


def main():
    MKT_DATE = datetime.date(2015, 4, 1)

    # swap = vanilla_swap_sched(datetime.date(2015, 4, 1))
    # swap_payments = vanilla_swap_payments(swap, datetime.date(2015, 6, 1))
    swap = VanillaSwap(MKT_DATE, MKT_DATE)
    print(swap.payments())
    print(swap.PV())


# main()
