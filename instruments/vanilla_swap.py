""" Prices simples EU vanilla swap.
"""
import datetime
import QuantLib as ql

from typing import List


def vanilla_swap_sched( start_date : datetime.date
                      , maturity   : str = '5Y'
                      , frequency  : str = '3M') -> ql.Schedule:
    """ Returns the schedule of the vanilla swap.

    """

    calendar = ql.TARGET()
    start    = ql.Date.from_date(start_date)
    maturity = calendar.advance(start, ql.Period(maturity))

    return ql.MakeSchedule(start, maturity, ql.Period(frequency))


def vanilla_swap_payments( swap_sched : ql.Schedule
                         , pricing_date : datetime.date ) -> List[datetime.date]:

    return [date.to_date() for date in swap_sched if date.to_date() >= pricing_date ]


def main():
    swap = vanilla_swap_sched(datetime.date(2015, 4, 1))
    swap_payments = vanilla_swap_payments(swap, datetime.date(2015, 6, 1))


# main()
