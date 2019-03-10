# constructs the lattice for the model
import config, numpy as np

import tolling
if config.CUDA_PRESENT: 
    import pycuda.cumath
    import pycuda.gpuarray as gpa


class LatticeLN(object):
    """
    Constructs a lattice for the log-normal model

    """

    def __init__( self
                , F0
                , sigma_F
                , sigma_C
                , nb_points
                , nb_std_dev = 2.
                , ci_ind     = False
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
        self.ci_ind = ci_ind
        self.construct_lattice()

    def construct_lattice(self):
        Z = np.linspace(- self.nb_std_dev, self.nb_std_dev, self.nb_points).astype(self.dtype_used)

        if self.ci_ind:
            self.lattice = self.F0 * pycuda.cumath.exp(self.sigma_C * gpa.to_gpu(Z))
        else:
            self.lattice = self.F0 * np.exp(self.sigma_C * Z)  # lattice grid

        return self.lattice

    def reconstruct_lattice(self, new_nb_points):
        self.nb_points = new_nb_points
        self.construct_lattice()


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
                , nb_std_dev=2.
                , ci_ind=False):
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
        self.ci_ind = ci_ind
        self.construct_lattice()

    def construct_lattice(self):
        """
        Constructs the log-normal lattice 2D
        """

        Z = np.linspace(- self.nb_std_dev, self.nb_std_dev, self.nb_points)

        if self.ci_ind:
            Z = gpa.to_gpu(Z).astype(np.float32)
            self.lattice = { 'power': self.F0[0] * pycuda.cumath.exp(self.sigma_C[0] * Z)   # power lattice
                           , 'gas'  : self.F0[1] * pycuda.cumath.exp(self.sigma_C[1] * Z) } # gas lattice
        else:
            self.lattice = { 'power': self.F0[0] * np.exp(self.sigma_C[0] * Z)   # power lattice
                           , 'gas'  : self.F0[1] * np.exp(self.sigma_C[1] * Z) } # gas lattice

        return self.lattice

    def reconstruct_lattice(self, new_nb_points):
        self.nb_points = new_nb_points
        self.construct_lattice()
