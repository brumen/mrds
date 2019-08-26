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

    def __init__( self
                , mktDate   : datetime.date
                , fwdName   : str
                , fwdTenors : List[datetime.date]
                , fwdValues : List[float]
                , dcf = 365.25 ):
        """


        """

        self._mktDate   = mktDate
        self._fwdName   = fwdName
        self._fwdTenors = fwdTenors
        self._fwdValues = fwdValues
        self._dcf       = dcf

        self.__fwdCurve = None
        self.__fwdTenorsNumeric = None

    def __relativeDate(self, fwdDate : datetime.date) -> float:
        return (fwdDate - self._mktDate).days / self._dcf

    @property
    def _fwdTenorsNumeric(self, dcf=365.25):
        """
        Converts datetime into numeric values given

        """

        if self.__fwdTenorsNumeric:
            return self.__fwdTenorsNumeric

        self.__fwdTenorsNumeric = [self.__relativeDate(fwdDate) for fwdDate in self._fwdTenors]
        return self.__fwdTenorsNumeric

    def _fwdCurve(self):
        """
        Constructs the splined forward curve.

        """

        if self.__fwdCurve:
            return self.__fwdCurve

        self.__fwdCurve = scipy.interpolate.splrep(self._fwdTenorsNumeric, self._fwdValues)
        return self.__fwdCurve

    def fwdValue(self, fwdDate) -> float:
        """
        Gets the forward value for the forward date.

        """

        if isinstance(fwdDate, list):  # TODO: list is problematic, as it can be list[float] or list[datetime.date]
            return scipy.interpolate.splev( [self.__relativeDate(fwdDates) for fwdDates in fwdDate]
                                          , self._fwdCurve)

        if isinstance(fwdDate, datetime.date):
            return scipy.interpolate.splev(self.__relativeDate(fwdDate), self._fwdCurve)

        if isinstance(fwdDate, float):
            return scipy.interpolate.splev(fwdDate, self._fwdCurve)

        raise FwdCurveException('Forward value {0} not recognized'.format(str(fwdDate)))

    def get_1nb( self, fwdDate  : datetime.date ):
        """
        Finds the 1st nearby contract to fwdDate.

        :param fwdDate: forward date to which the 1st nearby contract is searched.
        """

        if fwdDate > self._fwdTenors[-1]:
            raise FwdCurveException('Fwd date searched {0} larger than the curve last date {1}'.format(fwdDate, self._fwdTenors[-1]))

        largerDates = sum([ fwdTenor <= fwdDate for fwdTenor in self._fwdTenors])

        return self._fwdTenors[largerDates], self._fwdValues[largerDates]

    @classmethod
    def fromDB(cls, mktDate : datetime.date, fwdCurveName : str):
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
