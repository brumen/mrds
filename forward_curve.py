#
#  Constructs a forward curve object.
#

import datetime
import scipy.interpolate

from typing import List

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

        self._mkt_date   = mkt_date
        self.fwd_name   = fwd_name
        self.fwd_tenors = fwd_tenors
        self.fwd_values = fwd_values
        self._dcf       = dcf

        self.__fwdCurve = None
        self.__fwdTenorsNumeric = None

    def __relative_date(self, fwd_date : datetime.date) -> float:
        """

        """
        return (fwd_date - self._mkt_date).days / self._dcf

    @property
    def _fwdTenorsNumeric(self, dcf=365.25):
        """
        Converts datetime into numeric values given

        """

        if self.__fwdTenorsNumeric:
            return self.__fwdTenorsNumeric

        self.__fwdTenorsNumeric = [self.__relative_date(fwdDate) for fwdDate in self.fwd_tenors]
        return self.__fwdTenorsNumeric

    def _fwdCurve(self):
        """
        Constructs the splined forward curve.

        """

        if self.__fwdCurve:
            return self.__fwdCurve

        self.__fwdCurve = scipy.interpolate.splrep(self._fwdTenorsNumeric, self.fwd_values)
        return self.__fwdCurve

    def fwdValue(self, fwdDate) -> float:
        """
        Gets the forward value for the forward date.

        """

        if isinstance(fwdDate, list):  # TODO: list is problematic, as it can be list[float] or list[datetime.date]
            return scipy.interpolate.splev( [self.__relative_date(fwdDates) for fwdDates in fwdDate]
                                          , self._fwdCurve)

        if isinstance(fwdDate, datetime.date):
            return scipy.interpolate.splev(self.__relative_date(fwdDate), self._fwdCurve)

        if isinstance(fwdDate, float):
            return scipy.interpolate.splev(fwdDate, self._fwdCurve)

        raise FwdCurveException('Forward value {0} not recognized'.format(str(fwdDate)))

    def get_1nb( self, fwdDate  : datetime.date ):
        """
        Finds the 1st nearby contract to fwd_date.

        :param fwdDate: forward date to which the 1st nearby contract is searched.
        """

        if fwdDate > self.fwd_tenors[-1]:
            raise FwdCurveException('Fwd date searched {0} larger than the curve last date {1}'.format(fwdDate, self.fwd_tenors[-1]))

        largerDates = sum([fwdTenor <= fwdDate for fwdTenor in self.fwd_tenors])

        return self.fwd_tenors[largerDates], self.fwd_values[largerDates]

    @classmethod
    def from_db(cls, mktDate : datetime.date, fwdCurveName : str):
        """
        Reads the forward curves from the database.

        :param mktDate: market date
        :param fwdCurveName: name of the forward, e.g. 'WTI'
        """

        fwd_vol_matched = ds.read_data_matched_tenors(mktDate
                                                      , fwdCurveName
                                                      , fwdCurveName)

        return cls( mktDate
                  , fwdCurveName
                  , fwd_vol_matched['fwd_tenors_dt']
                  , fwd_vol_matched['fwd_curve'] )
