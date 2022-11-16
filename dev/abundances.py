import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt

from user_params import cosmo_params, physics_units, PBHForm

from power_spectrum import PowerSpectrum

class CLASSabundances:
    def __init__(self, 
                    powerspectrum=None, 
                    scaling=False,
                    gaussian=True):
        self.Pk = powerspectrum if powerspectrum else PowerSpectrum.default
        self.Pkrescaling = scaling
        self.Pkscalingfactor = scaling if isinstance(scaling, float) else 1.0 
        self.Gaussian = gaussian
        self.rescaling_is_done = True if isinstance(scaling, float) else False



    def get_logbeta(self, mPBH, use_thermal_history='default'):  #TODO

        if use_thermal_history == 'default':
            use_thermal_history = self.use_thermal_history

        if self.Pkrescaling == True and self.rescaling_is_done == False:
            self.calc_scaling(mPBH)

        PkS = self.Pk

        # returns the density fraction of PBH at formation \beta(m_PBH)
        mH = mPBH / self.ratio_mPBH_over_mH
        kk = self.kmsun / mH ** (0.5)  # S. Clesse: to be checked
        limit_for_using_erfc = 20.                    #TODO hardcoded
        Pofk = PkS.Pk(kk) * self.Pkscalingfactor

        # if verbose: print("323 :: ", Pofk, kk)

        zetacr = self.get_zetacr(mPBH,  use_thermal_history=use_thermal_history)

        if (self.Gaussian):
            argerfc = zetacr / (2. * Pofk) ** (1. / 2.)
            logbeta = np.zeros_like(argerfc)
            mask = (argerfc < limit_for_using_erfc)
            logbeta[mask] = np.log10(erfc(argerfc[mask]))
            logbeta[~mask] = -np.log10(argerfc[~mask] * np.sqrt(np.pi)) - argerfc[~mask] ** 2 / np.log(10.)


        else:
            raise (ValueError, "Non-Gaussian effects are not implemented yet")

        return logbeta

    def get_k(self, mPBH):
        # Returns the scale (in [Mpc^-1] )  for a given PBH mass
        mH = mPBH / self.ratio_mPBH_over_mH
        kk = self.kmsun * np.sqrt(1. / mH)  # S. Clesse: to check
        return kk

    def calc_scaling(self, mPBH):
        # Compute the Rescaling of the power spectrum to get the forcefPBH

        
        def _rootfunction_logfPBH(scaling):
            self.Pkscalingfactor = 10. ** scaling
            function = self.get_logfPBH(mPBH) - np.log10(self.forcedfPBH)
            return function

        if self.Pkrescaling == False:
            raise Exception('Error: rescaling of the power spectrum when not allowed (ie. Pkrescaling = False)')
        self.rescaling_is_done = True  # Needed to stop/avoid bucle

        if verbose > 1:
            print(":: Rescaling of the power spectrum to get f_PBH =", self.forcedfPBH)

        sol = opt.bisect(_rootfunction_logfPBH, -1., 1., rtol=1.e-5, maxiter=100)
        self.Pkscalingfactor = 10. ** sol

        if verbose > 1:
            print(":: After rescaling, I get a total abundance of PBHs: fPBH=", self.get_fPBH(mPBH))
            print(":: Rescaling factor=", self.Pkscalingfactor)
            print(":: zeta_crit (radiation) = ", self.get_deltacr())
            print("====")

        return self.Pkscalingfactor


    def get_nonrescaled_logfofmPBH(self, mPBH, mHeq = 2.8e17):   #TODO   mHeq is hardcoded
        # returns log_10 of the dark matter density fraction of PBH today f(m_PBH)
        cp = self.cp
        logbeta = self.get_logbeta(mPBH)
        logfofmPBH = logbeta + np.log10(
            (mHeq / (mPBH / self.ratio_mPBH_over_mH)) ** (1. / 2.) * 2. / (cp.Omc / (cp.Omc + cp.Omb)))
        return logfofmPBH

    def get_logfofmPBH(self, mPBH, mHeq = 2.8e17):   #TODO   mHeq is hardcoded
        # returns log_10 of the dark matter density fraction of PBH today f(m_PBH)
        if self.Pkrescaling == True and self.rescaling_is_done == False:
            self.calc_scaling(mPBH)
        logfofmPBH = self.get_nonrescaled_logfofmPBH(mPBH, mHeq)
        return logfofmPBH

    def get_fofmPBH(self, mPBH):
        return 10**self.get_logfofmPBH(mPBH)

    def get_fPBH(self, mPBH):

        def _integrator_foflogmPBH(logmPBH):
            # returns the dark matter density fraction of PBH today f(m_PBH)
            logfofmPBH = logfofmPBH_interp(logmPBH)
            foflogmPBH = 10. ** logfofmPBH
            return foflogmPBH

        logmPBHtable = np.log10(mPBH)
        logmass_min = np.min(logmPBHtable)
        logmass_max = np.max(logmPBHtable)
        logfofmPBHtable = self.get_nonrescaled_logfofmPBH(mPBH)

        # Integrate
        logfofmPBH_interp = interp1d(logmPBHtable, logfofmPBHtable)
        fPBHsol = integrate.quad(_integrator_foflogmPBH, logmass_min, logmass_max, epsrel=0.001)
        fPBH = fPBHsol[0]

        return fPBH

    def get_logfPBH(self, mPBH_table):
        logfPBH = np.log10(self.get_fPBH(mPBH_table))
        return logfPBH

    def get_Pk(self, k):
        return self.Pk(k)
