""" General trade structure.
"""

import datetime

from typing      import Dict, Optional
from mongoengine import Document, StringField, DictField, connect

from mrds.instruments.vanilla_swap import VanillaSwap


MKT_DATE = datetime.date(2015, 4, 1)  # IMPORTANT: THIS SHOULD BE THE DATE, OTHERWISE CURVE FAILS.

connect(alias='trades', db='trades')  # Need a Mongo db running, with trades database.


class Trade(Document):

    meta = {'db_alias': 'trades'}

    type_  = StringField(required=True)  # type of trade (VanillaSwap, or other)
    params = DictField()    # params used for the trade type
    name   = StringField(required=True)  # name of the trade

    __instr_cache = None  # cache for the object

    @property
    def _instrument_type(self):
        """ Returns the trade instrument depending on type_. Basically a collection of handled instruments.
        """

        if self.type_ == 'VanillaSwap':
            return VanillaSwap

        raise RuntimeError(f'Unhandled type {_type}')

    def _instrument(self, market_date : Optional[datetime.date] = None):
        """ returns the object given the market date and parameters.

        :param market_date: market date.
        """

        if self.__instr_cache is not None:
            return self.__instr_cache

        self.__instr_cache = self._instrument_type.from_db(market_date, self.params)  # IMPORTANT: Instruments have to implement from_db method
        return self.__instr_cache

    def PV(self, market_date = None):
        """ Computes the PV of the trade.

        :param market_date: market date, if None, use today as date.
        """

        market_date_used = datetime.date.today() if market_date is None else market_date

        return self._instrument(market_date_used).PV()


def main():
    # examples of trade
    trade_2 = Trade(type_ = 'VanillaSwap', params = {'start_date': datetime.date(2015, 4, 5).isoformat()}, name = 'trade_3')
    print(trade_2.PV(MKT_DATE))


# main()
