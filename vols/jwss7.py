# JWSS7 volatility structure
import datetime
import numpy as np

from vols.vols import Volatility


class JWSS7Volatility(Volatility):
    """ Jump-wing parametrization.
    """

    @property
    def _vol_name(self):
        return 'JWSS7'

    @property
    def _vol_dates(self):
        return 'vol_dates'

    def atmVol( self
              , fwdDate : datetime.date ) -> np.double :
        """
        Returns the atm forward for the fwd date fwd_date.

        :param fwdDate: forward date for which the ATM is constructed
        """

        return self._volParams['vol_curve'][self._vol_for_date(fwdDate)][0]  # first elt is atm vol.

    @lru_cache(maxsize=None)
    def _transform_from_jwss7(self, fwdDate : datetime.date ):
        """
        Returns jw7 parametrization from jwss7 for particular fwd date.

        vol_params in Jwss7: [S0, atm, skew, smile, putslope, putbend, callslope, callbend]
        vol_params in jw7  : [S0, atm, A   , B    , C       , P      , alphaC   , alphaP  ]
        """

        nthContract = self._vol_for_date(fwdDate)
        volParams   = self._volParams['vol_curve'][nthContract]

        sigma_0, skew, smile, put_slope, put_bend, call_slope, call_bend = volParams

        B = (2. * skew + put_slope) / (put_slope + call_slope)
        A = 0.5 * B * (1. - B) * (call_slope + put_slope)**2 / (smile + skew**2)

        return {'sigma_0': sigma_0,
                'A'      : A,
                'B'      : B,
                'C'      : call_slope / A,
                'P'      : put_slope / A,
                'alphaC' : call_bend,
                'alphaP' : put_bend }

    def _vol_compute(self, fwdUsed : datetime.date, z : np.double ):
        """
        Computes the volatility given the following parameters:

        :param z: normalized strike
        :param alphaC, alphaP, ...: parameters for the JW7 parametrization.

        """

        volParams = self._transform_from_jwss7(fwdUsed)

        return volParams['sigma_0'] * sqrt(1. + volParams['A'] * log(volParams['B'] * exp(volParams['C'] * (z / (1.0 + z * z) ** (volParams['alphaC']/2))) + \
                                                                     (1. - volParams['B']) * exp(- volParams['P'] * (z / (1.0 + z * z) ** (volParams['alphaP']/2)))))

    def implied_vol(self, fwdDate_ : datetime.date or int, K : np.double, ttm : np.double):
        """
        Implied vol for the fwd

        :param fwd: forward tenor - could be either an integer, like 5,
                    or datetime.date
        :type fwd: int or datetime.date
        :param K: strike price
        :param ttm: time to maturity
        """

        fwdUsed = fwdDate_ if isinstance(fwdDate_, int) else self._vol_for_date(fwdDate_)
        volParams = self._transform_from_jwss7(fwdDate_)

        _, fwdValues = self._fwd_params

        return self._vol_compute( fwdUsed
                                , JWSS7Volatility.normalized_strike(fwdValues[fwdUsed]
                                                                    , np.array([K])
                                                                    , volParams['sigma_0']
                                                                    , ttm) )

    def localVol(self, fwdDate : datetime.date, S, T, ttm):
        """
        Local volatility of the JWSS7 parametrization.

        :param fwdDate: forward index that we are computing the local vol of
        :param ttm: option time to maturity
        """

        jw7Params = self._transform_from_jwss7(fwdDate)
        sigma_0 = jw7Params['sigma_0']
        A       = jw7Params['A']
        B       = jw7Params['B']
        C       = jw7Params['C']
        P       = jw7Params['P']
        alphaC  = jw7Params['alphaC']
        alphaP  = jw7Params['alphaP']

        z = self.normalized_strike(S_0, S, sigma_0, ttm)  # TODO: CHECK HERE!!
        sigma = self.implied_vol(S, T)

        d1 = (log(S / S_0) + sigma * sigma * ttm / 2.0) / (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)
        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * sqrt (ttm) ) / ( sqrt ( 1.0 + A * log (Xz) ) ) * \
            (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S / K ) + sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S / K ) - sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        denomin = (sigma_0 * sqrt(ttm) * K * Xz * sqrt(1.0 + A * log(Xz)))
        BCexpr = (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        sigmaKK = A / (2.0 * sqrt(ttm)) * (- A / (2.0 * denomin * K * Xz * (1.0 + A * log(Xz))) *
                                           BCexpr * BCexpr - BCexpr * BCexpr / (denomin * K * Xz) +
                                           (B * C ** 2 * exp(C * z) + P ** 2 * (1.0 - B) * exp(- P * z)) /
                                           (denomin * K) - BCexpr *
                                           sigma_0 *
                                           sqrt(ttm) / (denomin * K)
                                           )

        # derivative of z wrt t
        zt = log(K / S) / sigma_0 * (-0.5 * ttm**(- 1.5))
        sigmat = sigma_0 ** 2 / (2.0 * sigma) * A / Xz  * \
            ( B * C * exp ( C * z) - P * (1 - B) * exp ( - P * z ) ) * \
            zt  # derivative of sigma wrt t

        up_part = sigma * sigma + 2.0 * ttm * sigma * sigmat

        down_part = (1.0 + K * d1 * sqrt (ttm) * sigmaK ) ** 2.0 + K * K * ttm * sigma * \
                    (sigmaKK - d1 * sigmaK * sigmaK * ttm)

        # catching nan-s
        if (up_part / down_part < 0.0):
            logger.info("Caution: Imaginary local vol., using ATM vol.")
            return sigma_0

        return sqrt(up_part / down_part)

    def callFutureT(self, fwdDate, S0, K, ttm):
        """
        Derivative of Black's call (with expirty time  T) on a futures contract
           with maturity ttm (cond: T < ttm)

        :param fwdDate: forward index that we are drawing the vol of
        :param S0:
        :param K: strike value
        :param ttm: time to maturity
        """

        jw7Params = self._transform_from_jwss7(fwdDate)
        sigma_0 = jw7Params['sigma_0']
        A       = jw7Params['A']
        B       = jw7Params['B']
        C       = jw7Params['C']
        P       = jw7Params['P']
        alphaC  = jw7Params['alphaC']
        alphaP  = jw7Params['alphaP']

        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(S0, K, ttm)
        S0_local = S0  # TODO: CHECK IF THIS IS REALLY NECESSARY

        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        d1 = (log(S0_local / K) + sigma * sigma * ttm / 2.0) / \
            (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)
        zt = log(K / S0_local) / sigma_0 * (-0.5 * pow(ttm, - 1.5))  # z wrt t
        sigmat = sigma_0 * sigma_0 / (2.0 * sigma) * A / Xz  * \
            ( B * C * exp ( C * z) - P * ( 1.0 - B) * exp ( - P * z ) ) * \
            zt  # derivative of sigma wrt t
        sigma2t = 2.0 * sigma * sigmat  # derivative of sigma^2 wrt t

        d1T = ((sigma2t * ttm / 2.0 + sigma * sigma / 2.0) * sigma * sqrt(ttm) -
               ( log ( S0_local / K) + sigma * sigma * ttm / 2.0 ) * ( sigmat * ttm + sigma / (2.0 * sqrt (ttm) ) ) ) \
            / (sigma * sigma * ttm)

        d2T = (- (sigma2t * ttm / 2.0 + sigma * sigma / 2.0) * sigma * sqrt(ttm) -
               ( log ( S0_local / K) - sigma * sigma * ttm / 2.0 ) * ( sigmat * ttm + sigma / (2.0 * sqrt (ttm) ) ) ) \
            / (sigma * sigma * ttm)

        return S0_local * \
            scipy.stats.norm.pdf(d1) * d1T - K * scipy.stats.norm.pdf(d2) * d2T

    # first derivative of (undiscounted) Black's call wrt K
    def call_future_K(self, fwd, S0, K, ttm):
        """
        Derivative of a call option in this parametrization wrt strike price K.

        """

        jw7Params = self._transform_from_jwss7(fwdDate)
        sigma_0 = jw7Params['sigma_0']
        A       = jw7Params['A']
        B       = jw7Params['B']
        C       = jw7Params['C']
        P       = jw7Params['P']
        alphaC  = jw7Params['alphaC']
        alphaP  = jw7Params['alphaP']

        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(S0, K, ttm)
        S0_local = S0  # TODO: CHECK IF THIS IS REALLY NECESSARY
        d1 = (log(S0_local / K) + sigma * sigma * ttm / 2.0) / \
            (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)

        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * sqrt (ttm) ) / sqrt ( 1.0 + A * log (Xz) ) * \
            (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S0_local / K ) + sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S0_local / K ) - sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        return (S0_local * scipy.stats.norm.pdf(d1) * d1K -
                scipy.stats.norm.cdf(d2) - K * scipy.stats.norm.pdf(d2) * d2K)

    # second derivative wrt K of the undiscounted call
    def call_future_KK(self, fwdDate, S0, K, ttm):


        jw7Params = self._transform_from_jwss7(fwdDate)
        sigma_0 = jw7Params['sigma_0']
        A       = jw7Params['A']
        B       = jw7Params['B']
        C       = jw7Params['C']
        P       = jw7Params['P']
        alphaC  = jw7Params['alphaC']
        alphaP  = jw7Params['alphaP']
        S0_local = S0

        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(S0, K, ttm)
        S0_local = S0  # TODO: CHECK IF THIS IS REALLY NECESSARY

        d1 = (log(S0 / K) + sigma * sigma * ttm / 2.0) / (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)
        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * sqrt (ttm) ) / ( sqrt ( 1.0 + A * log (Xz) ) ) * \
            (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S0 / K ) + sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S0 / K ) - sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        denomin = (sigma_0 * sqrt(ttm) * K * Xz * sqrt(1.0 + A * log(Xz)))
        BCexpr = (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        sigmaKK = A / (2.0 * sqrt(ttm)) * (
            - A / (2.0 * denomin * K * Xz * (1.0 + A * log(Xz))) * BCexpr * BCexpr -
            BCexpr * BCexpr / (denomin * K * Xz) +
            (B * C * C * exp(C * z) + P * P * (1.0 - B) * exp(- P * z)) /
            (denomin * K) -
            BCexpr * sigma_0 * sqrt(ttm) / (denomin * K))

        d1KK = ((1.0 / (K * K) + sigmaK * sigmaK * ttm + sigma * ttm * sigmaKK) * sigma * sqrt(ttm) -
                ( log ( S0 / K ) + sigma * sigma * ttm  / 2.0 ) * sqrt (ttm) * sigmaKK ) / ( sigma * sigma * ttm) - \
                  \
            2.0 * ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
                   ( log ( S0_local / K ) + sigma * sigma * ttm  / 2.0 ) * sqrt (ttm) * sigmaK ) * sigma * ttm * sigmaK \
            / (sigma * sigma * sigma * sigma * ttm * ttm)

        d2KK = ((1.0 / (K * K) - sigmaK * sigmaK * ttm - sigma * ttm * sigmaKK) * sigma * sqrt(ttm) -
                ( log ( S0 / K ) - sigma * sigma * ttm  / 2.0 ) * sqrt (ttm) * sigmaKK ) / ( sigma * sigma * ttm) - \
            2.0 * ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
                   ( log ( S0_local / K ) - sigma * sigma * ttm  / 2.0 ) * sqrt (ttm) * sigmaK ) * sigma * ttm * sigmaK \
            / (sigma * sigma * sigma * sigma * ttm * ttm)

        return (S0 * normpdfD(d1) * d1K * d1K + S0_local * scipy.stats.norm.pdf(d1) * d1KK -
                2.0 * scipy.stats.norm.pdf(d2) * d2K - K * scipy.stats.norm.pdf(d2) * (-d2) * d2K * d2K - K * scipy.stats.norm.pdf(d2) * d2KK)

    def skewed_distribution(self, fwdDate, S0, K, ttm):
        return 1. + self.call_future_K(fwdDate, S0, K, ttm)

    def inversion_skewed_cdf( self
                            , fwdDate
                            , S0
                            , ttm
                            , quantile
                            , lb = 0.01
                            , ub = np.inf
                            , maxIter = 150
                            , iprint  = -9 ):
        """ Function finds K such that: skewed_cdf_analy(K, quantile) = 0
        """

        return NLP( lambda K: (self.skewed_distribution(fwdDate, S0, K, ttm) - quantile)**2
                      , S0
                      , lb      = lb
                      , ub      = ub
                      , maxIter = maxIter
                      , iprint  = iprint ).solve('scipy_cobyla').xf[0]


class JWSS7VolatilityDisplay(JWSS7Volatility, VolatilityDrawMixin):

    def jw7_buttons(self, fwd, root, ax, dataPlot_canvas):
        fct_update = lambda cc: self.update_graph(fwd, model, array([c1.get(), c2.get(), c3.get(), c4.get(), c5.get(), c6.get(), c7.get(), c8.get()]), ax,
                                             dataPlot_canvas)
        # root ... Tk root
        # ax ... Axes3D object
        # dataPlot_canvas ... canvas object

        # parameter tk.SCALEs
        c1 = tk.Scale(root, from_=80.0, to=120.0, resolution=0.1, label="S0", orient=tk.HORIZONTAL,
                   command=fct_update)
        c2 = tk.Scale(root, from_=0.05, to=0.8, resolution=0.05, label="sig", orient=tk.HORIZONTAL,
                   command=fct_update)
        c3 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.25, label="A", orient=tk.HORIZONTAL,
                   command=fct_update)
        c4 = tk.Scale(root, from_=0.0, to=1.0, resolution=0.05, label="B", orient=tk.HORIZONTAL,
                   command=fct_update)
        c5 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.2, label="C", orient=tk.HORIZONTAL,
                   command=fct_update)
        c6 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.2, label="P", orient=tk.HORIZONTAL,
                   command=fct_update)
        c7 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.2, label="alpha_C", orient=tk.HORIZONTAL,
                   command=fct_update)
        c8 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.2, label="alpha_P", orient=tk.HORIZONTAL,
                   command=fct_update)

        c1.grid(row=0, column=1)
        c2.grid(row=1, column=1)
        c3.grid(row=2, column=1)
        c4.grid(row=3, column=1)
        c5.grid(row=4, column=1)
        c6.grid(row=5, column=1)
        c7.grid(row=6, column=1)
        c8.grid(row=7, column=1)

        # replot button
        b1 = tk.Button(root, text="replot", command=lambda: update_graph(fwd, model, array([c1.get(), c2.get(), c3.get(), c4.get(), c5.get(), c6.get(), c7.get(), c8.get()]), ax,
                                                                      dataPlot_canvas)).grid(row=8, column=0)

        dataPlot_canvas.show()
        root.mainloop()
