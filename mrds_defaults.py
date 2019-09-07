# Defaults of the ComSkew
import numpy as np

from functools import lru_cache


# TODO: THIS CLASS NEEDS TO BE REFACTORED.
class ComSkewDefaultsMixin:
    """ Defaults mixin class for Mrds model.
    """

    # TODO: THINK ABOUT maxsize= 100
    @lru_cache(maxsize=100)
    def _sigma_default(self, nbFactors: int, sigma_type='init'):
        """
        Defines the default value of the sigma parameters.

        :param sigmaType: either 'init', 'lb', 'ub'
        :param nbFactors: number of factors TODO: FINISH THIS PART HERE
        """

        sigmaDefault = { 'init': np.array([0.188, 0.101])
                       , 'lb'  : np.array([0.05, 0.01])
                       , 'ub'  : np.array([4., 1.]) }

        return sigmaDefault[sigma_type]

    @lru_cache(maxsize=100)
    def _kappa_default(self, nbFactors : int, kappa_type='init'):
        """ Defines the default value of the kappa parameters.

        :param nbFactors: number of factors to use in the calibration, choice of 2, 1, TODO: FINISH HERE!!!
        :param kappa_type: either 'kappa_init', 'kappa_lb', 'kappa_ub'
        """

        # TODO: INCORPORATE THE NUMBER OF FACTORS
        kappaDefault = { 'init': np.array([0.1, 0.5])
                       , 'lb'  : np.array([0.05, 0.01])
                       , 'ub'  : np.array([12., 1.]) }

        return kappaDefault[kappa_type]
