"""
Second order SGWB from inflationary primordial scalar powerspectrum
"""
import numpy as np
from scipy.integrate import dblquad
# import scipy.constants as const
# import scipy.special as special
# from scipy.special import erfc
# from scipy.interpolate import interp1d
# import scipy.integrate as integrate
# import scipy.optimize as opt

class SecondOrderSGWB():

    def init():
        pass

    # Functions to compute the SGWB from second order perturbations
    def IC2(self, d, s):
        return -36 * np.pi * (s ** 2 + d ** 2 - 2) ** 2 / (s ** 2 - d ** 2) ** 3 * np.heaviside(s - 1, 1)

    def IS2(self, d, s):
        return -36 * (s ** 2 + d ** 2 - 2) / (s ** 2 - d ** 2) ** 2 * (
                    (s ** 2 + d ** 2 - 2) / (s ** 2 - d ** 2) * np.log((1 - d ** 2) / np.absolute(s ** 2 - 1)) + 2)

    def IcsEnvXY(self, x, y):
        return (self.IC2(np.absolute(x - y) / (3 ** 0.5), np.absolute(x + y) / (3 ** 0.5)) ** 2 + self.IS2(
            np.absolute(x - y) / (3 ** 0.5), np.absolute(x + y) / (3 ** 0.5)) ** 2) ** 0.5

        # Integral returning the spectrum

    def compint(self, kvval, sigmaps):
        value, error = dblquad(lambda x, y:
                                x ** 2 / y ** 2 * (1 - (1 + x ** 2 - y ** 2) ** 2 / (4 * x ** 2)) ** 2
                                * self.Pk(kvval * self.mpc / self.c) * self.Pkscalingfactor  # PS(kvval*x)
                                * self.Pk(kvval * self.mpc / self.c) * self.Pkscalingfactor  # PS(kvval*y)
                                * self.IcsEnvXY(x, y) ** 2
                                ,
                                10 ** (- 4 * sigmaps) / kvval, 10 ** (4 * sigmaps) / kvval, lambda x: np.absolute(1 - x),
                                lambda x: 1 + x)
        return value

    
    def eval_oldcode(self):

        # Compute the SGWB from 2nd order perturbations
        print("Step 2:  Computation of the GW spectrum from denstiy perurbations at second order")
        print("Can take several minutes depending on the value of nk...")
        print("====")

        ks = self.kp / self.mpc * self.c
        sigmaps = 0.5
        print("ks = ", ks)

        kvals = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nk)
        # print self.Pk(kvals * self.mpc /self.c) * self.Pkscalingfactor
        # print "coucou 2nd order GW"
        kres = np.array([self.compint(xi, sigmaps) for xi in kvals])
        # coefficient due to thermal history see Eq. (2.11) https://arxiv.org/pdf/1810.12224.pdf
        # to be updated depending on the reference peak of the spectrum, to integrated with the rest of the code
        Omega_r_0 = 2.473 * 1e-5
        norm = self.ratio_mPBH_over_mH * Omega_r_0 / 972.

        #        self.k_2ndOmGW= ks*kvals
        self.freq_2ndOmGW = ks * kvals / 2. /np.pi
        # print "Second coucou 2nd order GW"
        self.OmGW_2ndOmGW = norm * kres