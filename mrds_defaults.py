# Defaults of the ComSkew
import numpy as np

from functools import lru_cache


class ComSkewDefaultsMixin:
    """ Defaults mixin class for Mrds model.
    """

    # TODO: THINK ABOUT maxsize= 100
    @lru_cache(maxsize=100)
    def _sigma_default(self, nb_factors: int, sigma_type='init'):
        """ Defines the default value of the sigma parameters.

        :param sigma_type: either 'init', 'lb', 'ub'
        :param nb_factors: number of factors TODO: FINISH THIS PART HERE
        """

        allowed_types = ('init', 'lb', 'ub', )
        assert sigma_type in allowed_types, f'Sigma type {sigma_type} not in {allowed_types}.'

        if sigma_type == 'init':
            return np.array([0.188, 0.101])

        if sigma_type == 'lb':
            return np.array([0.05, 0.01])

        if sigma_type == 'ub':
            return np.array([4., 1.])

    @lru_cache(maxsize=100)
    def _kappa_default(self, nb_factors : int, kappa_type='init') -> np.array:
        """ Defines the default value of the kappa parameters for initial guess,
            upper and lower boundary for kappa search.

        :param nb_factors: number of factors to use in the calibration, choice of 2, 1, TODO: FINISH HERE!!!
        :param kappa_type: either 'kappa_init', 'kappa_lb', 'kappa_ub'
        :returns: default kappa value for the particular type (initial guess, upper and lower boundary.)
        """

        allowed_types = ('init', 'lb', 'ub', )
        assert kappa_type in allowed_types, f'Kappa type {kappa_type} not in {allowed_types}.'

        # TODO: INCORPORATE THE NUMBER OF FACTORS
        if kappa_type == 'init':
            return np.array([0.1, 0.5])

        if kappa_type == 'lb':
            return np.array([0.05, 0.01])

        if kappa_type == 'ub':
            return np.array([12., 1.])
