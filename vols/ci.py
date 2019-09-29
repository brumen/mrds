

from vols.vols import Volatility


class CIVolatility(Volatility):
    """ CI parametrization.
    """

    def __init__(self, tenor_l, delta_mn_l, vols_l, omega_l):
        """
        tenor_l: tenors list [1., 2.]
        delta_mn: list of lists of delta in log-m: N(log-moneyn. log(F/F_0) / sigma / sqrt(T)
        vols: list of lists of vols for corr. moneyness
        omega_l: list of omegas, smoothing parameters
        """

        self.name = 'ci'
        self.nb_fwds = len(tenor_l)
        self.tenor_l = tenor_l
        self.delta_mn_l = delta_mn_l
        self.vols_l = vols_l
        self.omega_l = omega_l

        self.implied_vol = {}
        for tenor, (T, delta_mn, vol, omega) in enumerate(zip(self.tenor_l, self.delta_mn_l,
                                                              self.vols_l, self.omega_l)):
            self.implied_vol[tenor] = lambda z: self.convInterp(delta_mn, vol, omega, z)

    def gen_impl_surf(self, fwd, ttm_grid, delta_grid):
        """
        generates impl. vols surface for fwd(scalar) and K_grid(v), ttm
        """
        return self.implied_vol[0](delta_grid)  # WRONG WRONG - 0 HERE

    def Phi(self, n, x, xx, omega):
        return norm.cdf((xx[n]-x)/omega)

    def phi(self, n, x, xx, omega):
        return norm.pdf((xx[n]-x)/omega)

    def J(self, x, xx, omega):
        JJ = np.array([])
        tmp = (1.-(x - xx[0])/(xx[1] - xx[0]))*(self.Phi(1, x, xx, omega) - self.Phi(0, x, xx, omega))
        tmp += (omega/(xx[1]-xx[0]))*(self.phi(1, x, xx, omega) - self.phi(0, x, xx, omega))
        tmp += (1.-norm.cdf((x-xx[0])/omega))
        JJ = np.append(tmp, JJ)
        N = len(xx)-1

        for n in range(1, N):
            tmp1 = (1.-(x-xx[n])/(xx[n+1]-xx[n]))*(self.Phi(n+1, x, xx, omega)-self.Phi(n, x, xx, omega))
            tmp1 += (omega/(xx[n+1]-xx[n]))*(self.phi(n+1, x, xx, omega)-self.phi(n, x, xx, omega))
            tmp2 = (x-xx[n-1])/(xx[n]-xx[n-1])*(self.Phi(n, x, xx, omega)-self.Phi(n-1, x, xx, omega))
            tmp2 -= (omega/(xx[n]-xx[n-1]))*(self.phi(n, x, xx, omega)-self.phi(n-1, x, xx, omega))
            JJ = np.append(JJ, tmp1 + tmp2)

        tmp = (x-xx[-2])/(xx[-1]-xx[-2])*(self.Phi(N, x, xx, omega) - self.Phi(N-1, x, xx, omega))
        tmp -= (omega/xx[-1]-xx[-2])*(self.phi(N, x, xx, omega)-self.phi(N-1, x, xx, omega))
        tmp += norm.cdf((x-xx[-1])/omega)
        JJ = np.append(JJ, tmp)

        return JJ

    def I(self, x, xx, omega):
        II = np.zeros([len(xx), len(xx)])
        tmp = 1. - norm.cdf((x-xx[0])/omega)
        II00 = tmp
        N = len(xx) - 1
        for n in range(N):
            tmp1 = (1.-(x-xx[n])/(xx[n+1]-xx[n])) * (self.Phi(n+1, x, xx, omega) - self.Phi(n, x, xx, omega))
            tmp1 += (omega/(xx[n+1]-xx[n])) * (self.phi(n+1, x, xx, omega) - self.phi(n, x, xx, omega))
            II[n, n] += tmp1
            tmp2 = (x-xx[n])/(xx[n+1]-xx[n]) * (self.Phi(n+1, x, xx, omega) - self.Phi(n, x, xx, omega))
            tmp2 -= (omega/(xx[n+1]-xx[n])) * (self.phi(n+1, x, xx, omega) - self.phi(n, x, xx, omega))
            II[n+1, n] = II[n+1, n] + tmp2
        IInn = norm.cdf((x-xx[-1])/omega)
        return II, II00, IInn

    def convInterp(self, lm_v, vol_v, omega, lm_new_v):
        """
        convolution interpolation:
        inputs:
          (lm_v, vol_v) .. pairs of log-moneyness, vol
          ln_new_v ... log-m where you want to compute vols
          omega ... parameter
        """
        if (type(lm_new_v) is not np.ndarray) and (type(lm_new_v) is not list):
            lm_new_v_arr = [lm_new_v]
        else:
            lm_new_v_arr = lm_new_v
        lm_new_v_len = len(lm_new_v_arr)
        lm_v_len = len(lm_v)
        Jnm = np.zeros([lm_v_len, lm_v_len])
        for ii in range(lm_v_len):
            Jnm[ii, :] = self.J(lm_v[ii], lm_v, omega)
        # CONTINUE HERE

    def extract_ind(self, p_mat):
        """
        extracts the model parameters
        """
        self.name = 'ci'  # adds the name of the model
        self.sigma_0 = p_mat[:, 0]
        self.rr_25 = p_mat[:, 1]  # vector of rr_25 marks
        self.wg_25 = p_mat[:, 2]  # vector of wg
