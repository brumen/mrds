#
#  Constructs a forward curve object.
#

import datetime
import scipy.interpolate

from typing import List, Tuple

import ds


class FwdCurveException(Exception):
    pass


class FwdCurve:
    """ Forward curve object
    """

    def __init__(self
                 , mkt_date   : datetime.date
                 , fwd_name   : str
                 , fwd_tenors : List[datetime.date]
                 , fwd_values : List[float]
                 , dcf = 365.25):
        """ Forward curve class

        :param mkt_date: market date
        :param fwd_name: forward name of the curve
        :param fwd_tenors: tenors on the curve
        :param fwd_values: values for the respective tenors
        :param dcf: day-count factor.
        """

        self.mkt_date   = mkt_date
        self.fwd_name   = fwd_name
        self.fwd_tenors = fwd_tenors
        self.fwd_values = fwd_values
        self._dcf       = dcf

        self.__fwd_curve = None
        self.__fwd_tenors_numeric = None

    def __relative_date(self, fwd_date : datetime.date) -> float:
        """ Computes the numeric date between fwd_date and self.mkt_date

        :param fwd_date: date from which the numerical distance is computed to mkt_date
        :returns: numerical distance to mkt date.
        """

        return (fwd_date - self.mkt_date).days / self._dcf

    @property
    def _fwdTenorsNumeric(self, dcf=365.25):
        """
        Converts datetime into numeric values given

        """

        if self.__fwd_tenors_numeric:
            return self.__fwd_tenors_numeric

        self.__fwd_tenors_numeric = [self.__relative_date(fwdDate) for fwdDate in self.fwd_tenors]
        return self.__fwd_tenors_numeric

    def _fwd_curve(self):
        """ Constructs the splined forward curve internally.

        """

        if self.__fwd_curve:
            return self.__fwd_curve

        self.__fwd_curve = scipy.interpolate.splrep(self._fwdTenorsNumeric, self.fwd_values)
        return self.__fwd_curve

    def fwd_value(self, fwd_date) -> float:
        """ Gets the forward value for the forward date fwd_date

        """

        if isinstance(fwd_date, list):  # TODO: list is problematic, as it can be list[float] or list[datetime.date]
            return scipy.interpolate.splev([self.__relative_date(fwd_dates) for fwd_dates in fwd_date]
                                           , self._fwd_curve)

        if isinstance(fwd_date, datetime.date):
            return scipy.interpolate.splev(self.__relative_date(fwd_date), self._fwd_curve)

        if isinstance(fwd_date, float):
            return scipy.interpolate.splev(fwd_date, self._fwd_curve)

        raise FwdCurveException('Forward value {0} not recognized'.format(str(fwd_date)))

    def get_1nb(self, fwd_date  : datetime.date) -> Tuple[datetime.date, float]:
        """ Finds the 1st nearby contract to fwd_date.

        :param fwd_date: forward date to which the 1st nearby contract is searched.
        """

        if fwd_date > self.fwd_tenors[-1]:
            raise FwdCurveException('Fwd date searched {0} larger than the curve last date {1}'.format(fwd_date, self.fwd_tenors[-1]))

        largerDates = sum([fwdTenor <= fwd_date for fwdTenor in self.fwd_tenors])

        return self.fwd_tenors[largerDates], self.fwd_values[largerDates]

    @classmethod
    def from_db(cls, mkt_date : datetime.date, fwd_curve_name : str):
        """ Reads the forward curves from the database.

        :param mkt_date: market date
        :param fwd_curve_name: name of the forward, e.g. 'WTI'
        :returns: FwdCurve object for market date and curve.
        """

        fwd_vol_matched = ds.read_data_matched_tenors( mkt_date
                                                     , fwd_curve_name
                                                     , fwd_curve_name)

        return cls(mkt_date
                   , fwd_curve_name
                   , fwd_vol_matched['fwd_tenors_dt']
                   , fwd_vol_matched['fwd_curve'])
