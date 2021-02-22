# Com Skew w/ ORM database queries

import datetime
import numpy as np
import json

from logging import getLogger
from typing  import List, Callable

from mrds.mrds          import ComSkew
from mrds.vols.vols     import Volatility
from mrds.forward_curve import FwdCurve


from mrds.comskew_params import ComSkewLnParams, ComSkewCParams, create_session, DB


logger = getLogger(__name__)


class ComSkewORM(ComSkew):
    """ Commodity skew model with ORM baked in.
    """

    def __init__(self
                 , mkt_date       : datetime.date
                 , fwd_curves     : List[FwdCurve]
                 , vol_curves     : List[Volatility]
                 , discount_curve : Callable = None
                 , calc_date      : datetime.date = None
                 , dcf            : float = 365.25 ):

        """ Initialization of the skew model.

        :param mkt_date: market date
        :param fwd_curves: dictionary, where keys are fwd curve names ('WTI') and values are FwdCurve objects
                     forward curve names to be used in the model, e.g. ['WTI', 'BRENT']
        :param vol_curves: commodity vol curves, in case they are different than forward curves.
        :param discount_curve: discount curve, a function of fwd_date, returns lambda fwd_date: discount(mkt_date, fwd_date)
        :param calc_date: calculation date.
        :param dcf: day-count factor for computing numerical dates from actual.
        """

        super().__init__(mkt_date, fwd_curves, vol_curves, discount_curve = discount_curve, calc_date = calc_date, dcf = dcf)

        self.__db_session = create_session(DB)  # this is lazy evaluated

    def _kappa_vec(self, asset : str) -> np.ndarray:
        """ Holds the kappa vector for a particular asset.

        :param asset: asset to compute the kappa vector of (e.g. 'WTI')
        """

        if asset in self._kappa_vec_val:
            return self._kappa_vec_val[asset]

        # check database if this is stored anywhere
        kappa_db_results = self.__db_session.query(ComSkewLnParams).filter_by( market_date=self.mkt_date
                                                                             , commodity = asset
                                                                             , param     = 'kappa' ).all()

        if kappa_db_results:  # this is not-empty
            self._kappa_vec_val[asset] = np.array(kappa_db_results[0].val)  # converting back to array (from list)
            return self._kappa_vec_val[asset]  # returning

        # not found the value anywhere, store the value, and return it.
        res_kappa = super()._kappa_vec(asset)  # this also stores value
        self.__db_session.add(ComSkewLnParams( commodity   = asset
                                              , market_date = self.mkt_date
                                              , param       = 'kappa'
                                              , value       = json.dumps(res_kappa.tolist())))
        self.__db_session.commit()
        return res_kappa

    def _sigma_vec(self, asset : str):
        """ Calibrated sigma vector, depends on the _kappa_sigma_rho above.

        :param asset: asset to calculate sigma vector over.
        """

        if asset in self._sigma_vec_val:
            return self._sigma_vec_val[asset]

        # check database if this is stored anywhere
        sigma_db_results = self.__db_session.query(ComSkewLnParams).filter_by( market_date=self.mkt_date
                                                                                      , commodity = asset
                                                                                      , param     = 'sigma' ).all()

        if sigma_db_results:  # this is not-empty
            self._sigma_vec_val[asset] = np.array(sigma_db_results[0].val)
            return self._sigma_vec_val[asset]

        res_sigma = super()._sigma_vec(asset)
        self.__db_session.add(ComSkewLnParams( commodity   = asset
                                             , market_date = self.mkt_date
                                             , param       = 'sigma'
                                             , value       = json.dumps(res_sigma.tolist())))
        self.__db_session.commit()
        return res_sigma

    def _c_vec(self, asset : str, fwd_date : datetime.date) -> np.array:
        """ Returns the C vector (skew vector) for the asset and forward date.

        :param asset: asset for which C vector
        :param fwd_date: forward date.
        """

        # check db for this instance
        C_db = self.__db_session.query(ComSkewCParams).filter_by( commodity   = asset
                                                                , market_date = self.mkt_date
                                                                , fwd_date    = fwd_date ).all()

        if C_db:  # C_db is not empty
            if asset not in self._C_vec:
                self._C_vec[asset] = {fwd_date: np.array(C_db[0].val)}
                return self._C_vec[asset][fwd_date]

            if fwd_date not in self._C_vec[asset]:
                self._C_vec[asset][fwd_date] = np.array(C_db[0].val)
                return self._C_vec[asset][fwd_date]

        # C_db is not found, call the superclass routine.
        res_c = super()._c_vec(asset, fwd_date)  # this also stores values.
        self.__db_session.add(ComSkewCParams( commodity   = asset
                                            , market_date = self.mkt_date
                                            , fwd_date    = fwd_date
                                            , value       = json.dumps(res_c.tolist())))
        self.__db_session.commit()
        return res_c

    def _c_vec_calibrate( self
                         , asset            : str
                         , fwd_dates        : List[datetime.date] ):
        """ Calibrates the dates that are not yet calibrated for the asset.

        :param asset: the asset to calibrate, such as 'wti'
        :param fwd_dates: forward dates for which to calibrate
        """

        if asset in self._C_vec:
            already_calibrated_dates = set(self._C_vec[asset].keys())
            to_be_calibrated = set(fwd_dates).difference(already_calibrated_dates)
        else:
            to_be_calibrated = fwd_dates

        # check which of the to_be_calibrated are in the DB

        calibrated_in_db = {skew_C_obj.fwd_date: skew_C_obj.val
                            for skew_C_obj in self.__db_session.query(ComSkewCParams).filter_by( commodity   = asset
                                                                                               , market_date = self.mkt_date).all()}

        # to be added but just here
        if asset not in self._C_vec:
            self._C_vec[asset] = {}

        for fwd_date in set(to_be_calibrated).intersection(list(calibrated_in_db.keys())):
            self._C_vec[asset][fwd_date] = np.array(calibrated_in_db[fwd_date])

        self._c_vec_calibrate_force(asset, set(to_be_calibrated).difference(list(calibrated_in_db.keys())))
