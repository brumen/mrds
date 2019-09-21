# constructs the lattice for the model
import numpy as np


class LatticeLN:
    """ Constructs a lattice for the log-normal model
    """

    def __init__( self
                , F0
                , sigma_F
                , sigma_C
                , nb_points
                , nb_std_dev = 2.
                , dtype_used = np.float):
        """

        :param nb_std_dev: how many std. dev. of the lattice does it construct
        :param ci_ind: cuda indicator

        """

        self.dtype_used = dtype_used
        self.F0 = F0
        self.sigma_C = sigma_C
        self.sigma_F = sigma_F
        self.nb_points = nb_points
        self.nb_std_dev = nb_std_dev

    def construct_lattice(self):
        Z = np.linspace(- self.nb_std_dev, self.nb_std_dev, self.nb_points).astype(self.dtype_used)
        return  self.F0 * np.exp(self.sigma_C * Z)  # lattice grid

    def reconstruct_lattice(self, new_nb_points):
        self.nb_points = new_nb_points
        self.construct_lattice()


class LatticeLNCuda(LatticeLN):

    def construct_lattice(self):
        import pycuda.cumath
        import pycuda.gpuarray as gpa

        Z = np.linspace(- self.nb_std_dev, self.nb_std_dev, self.nb_points).astype(self.dtype_used)
        return self.F0 * pycuda.cumath.exp(self.sigma_C * gpa.to_gpu(Z))


class LatticeLN2D(object):
    """
    Constructs a 2d-lattice of 2 dimensional

    """

    def __init__( self
                , F0
                , sigma_F
                , sigma_C
                , rho
                , nb_points
                , nb_std_dev=2. ):
        """
        F0 ... 2d vector of initial forward prices,
        sigma_F ... 2d vector of vols
        sigma_C ... 2d vector of cash-vols
        rho ... forward correlation.
        nb_std_dev ... how many std. dev. of the lattice does it construct
        """

        self.F0 = F0
        self.sigma_C = sigma_C
        self.sigma_F = sigma_F
        self.nb_points = nb_points
        self.nb_std_dev = nb_std_dev

    def construct_lattice(self):
        """
        Constructs the log-normal lattice 2D
        """

        Z = np.linspace(- self.nb_std_dev, self.nb_std_dev, self.nb_points)

        return { 'power': self.F0[0] * np.exp(self.sigma_C[0] * Z)   # power lattice
               , 'gas'  : self.F0[1] * np.exp(self.sigma_C[1] * Z) } # gas lattice

    def reconstruct_lattice(self, new_nb_points):
        # TODO: THIS HERE IS BAD, REDO!
        self.nb_points = new_nb_points
        return self.construct_lattice()


class LatticeLN2DCuda(LatticeLN2D):

    def construct_lattice(self):
        """
        Constructs the log-normal lattice 2D
        """
        import pycuda.cumath
        import pycuda.gpuarray as gpa

        Z = gpa.to_gpu(np.linspace(- self.nb_std_dev, self.nb_std_dev, self.nb_points)).astype(np.float32)
        return { 'power': self.F0[0] * pycuda.cumath.exp(self.sigma_C[0] * Z)    # power lattice
               , 'gas'  : self.F0[1] * pycuda.cumath.exp(self.sigma_C[1] * Z) }  # gas lattice
