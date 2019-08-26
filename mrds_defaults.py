# Defaults of the ComSkew
import numpy as np


# TODO: THIS CLASS NEEDS TO BE REFACTORED.
class ComSkewDefaultsMixin:
    """ Defaults mixin class for Mrds model.
    """

    @property
    def sigmaPropertyMap(self):
        """ Map between sigma keywords and sigma properties.
        """

        return { 'init': self.__sigmaInit
               , 'lb'  : self.__sigmaLB
               , 'ub'  : self.__sigmaUB }

    @property
    def kappaPropertyMap(self):
        """
        Map between kappa keywords and kappa properties, used only internally in the class.

        """

        return { 'init': self.__kappaInit
               , 'lb'  : self.__kappaLB
               , 'ub'  : self.__kappaUB }

    def sigmaDefault(self, nbFactors: int, sigmaType='init'):
        """
        Defines the default value of the sigma parameters.

        :param sigmaType: either 'init', 'lb', 'ub'
        :param nbFactors: number of factors TODO: FINISH THIS PART HERE
        """

        sigmaDefault = { 'init': np.array([0.188, 0.101])
                       , 'lb'  : np.array([0.05, 0.01])
                       , 'ub'  : np.array([4., 1.]) }

        if self.sigmaPropertyMap[sigmaType]:
            return self.sigmaPropertyMap[sigmaType]

        self.sigmaPropertyMap[sigmaType] = sigmaDefault[sigmaType]
        return self.sigmaPropertyMap[sigmaType]

    def kappaDefault(self, nbFactors : int, kappaType='init'):
        """ Defines the default value of the kappa parameters.

        :param nbFactors: number of factors to use in the calibration, choice of 2, 1, TODO: FINISH HERE!!!
        :param kappaType: either 'kappa_init', 'kappa_lb', 'kappa_ub'
        """

        # TODO: INCORPORATE THE NUMBER OF FACTORS
        kappaDefault = { 'init': np.array([0.1, 0.5])
                       , 'lb'  : np.array([0.05, 0.01])
                       , 'ub'  : np.array([12., 1.]) }

        if self.kappaPropertyMap[kappaType]:
            return self.kappaPropertyMap[kappaType]

        self.kappaPropertyMap[kappaType] = kappaDefault[kappaType]
        return self.kappaPropertyMap[kappaType]

    def setSigmaDefault(self, sigmaType, sigmaNewValue):
        self.sigmaPropertyMap[sigmaType] = sigmaNewValue

    def setKappaDefault(self, kappaType, kappaNewValue):
        self.kappaPropertyMap[kappaType] = kappaNewValue
