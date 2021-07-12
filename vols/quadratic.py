
import datetime
import numpy as np

from tkinter import Scale, Button, HORIZONTAL
from typing  import List, Dict, Tuple

import mrds.ds as ds
from mrds.forward_curve import FwdCurve
from mrds.vols.vols import Volatility, VolatilityDrawMixin


class QuadraticVol(Volatility):
    """ Quadratic volatility parametrization, i.e.
        _smooth_ind is the smoothness indicator
        _alpha is the smoothness factor
    """

    def __init__(self
                 , com_name : str
                 , mkt_date : datetime.date
                 , fwd_params : FwdCurve
                 , vol_params : Dict[datetime.date, List]
                 , ):

        super(QuadraticVol, self).__init__(com_name, mkt_date, fwd_params=fwd_params, vol_params=vol_params)
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

    def _get_params(self, ttm : datetime.date) -> Tuple[float, float, float, float, float]:
        """ Gets the parameters for the time to maturity.

        :param ttm: time-to-maturity
        :returns:

        """

        next_date = self._get_next_date(ttm)

        return self._vol_params[next_date]

    def implied_vol(self, fwd_value : float, ttm : datetime.date, smooth_ind = True) -> float:
        """ Computes the implied volatility of quadratic volatility surface.

        :param fwd_value: forward value for which to compute
        :param ttm: time-to-maturity
        :param smooth_ind: indicator whether to smooth the curve.
        :returns: implied volatility for the parameters indicated.
        """

        atm_strike = self._fwd_params.fwd_value(ttm)
        z = np.log(fwd_value / atm_strike)

        c0, c1, c2, theta, alpha = self._get_params(ttm)

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


class QuadraticVolDraw(QuadraticVol, VolatilityDrawMixin):
    """ c0c1c2 vol class w/ the ability to draw implied vol surfaces.
    """

    def _update_graph( self
                     , fwd_date : datetime.date
                     , ttm      : datetime.date
                     , c_vec : Tuple[float, float, float, float] ):

        # TO UPDATE THE PARAMETERS.

        return self.implied_surf(fwd_date, ttm_grid, K_grid)

    def _draw_buttons(self, root, ax, dataPlot_canvas):
        """ Draws the buttons of the C0C1C2 volatility.

        """
        fct_update = lambda cc: self._update_graph( fwd_date
                                                 , model
                                                 , [c0.get(), c1.get(), c2.get(), alpha.get()]
                                                 , ax
                                                 , dataPlot_canvas )

        # parameter scales
        c0 = Scale( root
                     , from_      = 80.0
                     , to         = 120.0
                     , resolution = 0.1
                     , label      = 'c0'
                     , orient     = HORIZONTAL
                     , command    = fct_update )

        c1 = Scale(root, from_=0.05, to=0.8, resolution=0.05, label="c1", orient=HORIZONTAL,command=fct_update)
        c2 = Scale(root, from_=0.0, to=5.0, resolution=0.25, label="c2", orient=HORIZONTAL,command=fct_update)
        alpha = Scale(root, from_=0.0, to=1.0, resolution=0.05, label="_alpha", orient=HORIZONTAL,command=fct_update)

        c0.grid(row=0, column=1)
        c1.grid(row=1, column=1)
        c2.grid(row=2, column=1)
        alpha.grid(row=3, column=1)

        # replot button
        b1 = Button( root
                   , text = "replot"
                   , command = lambda: self.update_graph( fwd
                                                        , model
                                                        , [c0.get(), c1.get(), c2.get(), alpha.get()]
                                                        , ax
                                                        , dataPlot_canvas ) ).grid(row=8, column=0)
        dataPlot_canvas.show()
        root.mainloop()
