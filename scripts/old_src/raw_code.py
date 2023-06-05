import numpy as np
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt
import os
import math
import matplotlib as mp
import matplotlib.pyplot as plt
# import classy
from numpy import ma
from matplotlib import ticker, cm

prjdir = "/home/cjoana/git/LISA-PBH-WG/"
datadir = prjdir + "data/"
figdir = prjdir + "figures"
zetacr_file = "zetacr.dat"


class PrimBHoles:

    def __init__(self):

        # units and constants
        self.c = 2.997924e8  # (*Speed of light, [m/s] *)
        self.mpc = 3.085678e22  # (* Megaparsec [m] *)
        self.pc = 3.086e16  # (*parsec [m]*)
        self.G = 6.67428e-11  # (* Gravitational constant [m^3/kg/s^2] *)
        self.msun = 1.989e30  # (* Sun mass [kg]*)
        self.year = 365.25 * 24 * 3600  # (* year in seconds[s] *)
        self.hbar = 6.62607e-34 / (2. * math.pi)  # (* reduced planck constant in m^2 kg /s*)
        self.hp = 6.582e-16  # (* Planck constant in eV s *)
        self.AU = 1.4960e11  # (* Astronomical unit [m]*)
        self.kb = 8.617343e-5  # (* Boltzmann constant [eV /K] *);
        self.eVinJ = 1.60217e-19  # (*eV in Joule *)
        self.lplanck = 1.61e-35  # (*Planck length*)
        self.rhoplanck = 5.155e96  # (*Planck energy*)
        self.mplanck = 1.22e19  # (*Planck mass [eV]*)

        # Standard cosmological parameters
        self.ns = 0.961
        self.As = 2.1e-9
        self.Omb = 0.0456
        self.Omc = 0.245
        self.h = 0.7
        self.Nur = 3.046
        self.TCMB = 2.726
        self.kstar = 0.05

        self.H0 = self.h * 100000. / self.mpc  # (* Hubble rate today, [s^-1] *)
        self.rhocr = 3. * self.H0 ** 2 / (8. * math.pi * self.G)  # (* Critical Density [kg/m^3] *)
        self.ar = 7.5657e-16  # (* Stephan's constant in J m^-3 K^-4 *);
        self.Omr = self.ar * self.TCMB ** 4 / self.rhocr / self.c ** 2  # (* Density of photons *)
        self.Omnu = self.Omr * 7. / 8 * self.Nur * (4. / 11.) ** (4. / 3.)  # (* Density of neutrinos *)
        self.OmLambda = 1. - (self.Omc + self.Omb + self.Omr + self.Omnu)  # (* Density of Dark Energy *)

        # =====================================================================
        # Model for the primordial power spectrum of curvature fluctuations:
        # =====================================================================

        # "Powerlaw" model:
        # self.Pk_model = "powerlaw"
        # self.ktrans = 1.e3  # Scale of the transition between CMB amplitude and PBH amplitude
        # self.nsPBH = 0.97  # Spectral index
        # self.kp = 2.e6  # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
        # self.AsPBH = 0.0205  # Power spectrum amplitude at the reference scale kp.

        # "Log-normal":
        self.Pk_model = "lognormal"
        self.kp = 2.e6      # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
        self.AsPBH = 0.0205 # Power spectrum amplitude at the reference scale kp.
        self.sigma = 1. # Power spectrum amplitude at the reference scale kp.

        # "Gaussian":
        # self.Pk_model = "gaussian"
        # self.kp = 2.e6      # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
        # self.AsPBH = 0.0205 # Power spectrum amplitude at the reference scale kp.
        # self.sigma = 1. # Power spectrum amplitude at the reference scale kp.

        # "Broken power-law":
        # self.Pk_model = "broken_powerlaw"
        # self.kp = 2.e6      # Reference scale [mpc^-1] (2.e6 corresponds to mPBH = 1 Msun)
        # self.AsPBH = 0.0205 # Power spectrum amplitude at the reference scale kp.
        # self.nslow = 3.
        # self.nshigh = -0.5

        # "Power spectrum from preheating"
        # self.Pk_model = "preheating"
        # TBC with Theodoros's code

        # "Power from a file provided by the user
        # self.Pk_model = "from_file"
        # TBC with an example of file

        # ======================================================
        # Model for the distrubtion of curvature fluctuations
        # ======================================================

        # Gaussian perturbations:
        self.Gaussian = True

        # Non Gaussian perturbations:
        # self.Gaussian = False

        # self.NGmodel = "qudiff"

        # ===========================================
        # Model of PBH formation
        # ===========================================

        # self.gamma = 0.8
        self.ratio_mPBH_over_mH = 0.8  # Ratio between PBH and Hubble masses at formation
        self.kmsun = 2.1e6

        self.PBHform_model = "standard"
        self.zetacr_rad = 1.02

        # self.PBHform_model="Musco20"

        self.Pkrescaling = True  # option to rescale the power spectrum to get a fixed DM fraction
        self.forcedfPBH = 1.  # Imposed DM fraction made of PBHs
        self.Pkscalingfactor = 1.

        self.use_thermal_history = True  # option to include the effect of the equation-of-state changes due to the known thermal history
        self.data_directory = datadir
        self.zetacr_thermal_file = "zetacr.dat"  # File of the evolution of zeta_cr with thermal history
        self.zetacr_thermal_rad = 1.02  # Reference value of zeta_cr for this file

        # ===================================
        # Model of merging rates
        # ===================================

        # Primordial binaries
        self.merging_want_primordial = True
        self.fsup = 0.0025  # Rate suppression factor from N-body simulations when f_PBH > 0.1

        # Tidal capture in clusters
        self.merging_want_clusters = True
        self.Rclust = 400.

        # Hyperbolic encounters

        # Karsten's  model

        # self.formation_model = "std"
        # self.binary_model = "CGB16"
        # self.stochastic_model = "CGB17"
        # self.massfunction_model = "std"

        # ===================================
        # Precision/working parameters
        # ===================================

        self.logmmin = -6.  # Minimum log_10(PBH mass)
        self.logmmax = 8.  # Maximum log_10(PBH mass)
        self.Nmass = 1000  # Number of sampling points
        # self.dlogmass = 0.05

        # SGWB from second oder perturbaions
        # Set up spectrum limits (kmin, kmax) and number of points nk
        # wavenumbers are normalised with respect to the central kp value of the power spectrum
        self.kmin = 1e-3
        self.kmax = 1e1
        self.nk = 2

        # parameters for the redshift evolution
        self.zmin = 0.
        self.zmax = 5.
        # self.dz = 0.1
        self.Nz = 50

    def rhoCDM(self, a):
        # Returns the CDM density at scale factor a
        rhoCDM = self.Omc * self.rhocr / a ** 3
        return rhoCDM

    def rhob(self, a):
        # Returns the baryon density at scale factor a
        rho = self.Omb * self.rhocr / a ** 3
        return rho #rhoCDM

    def H(self, a):
        # Returns the Hubble rate parameter (in s^-1) at scale factor a
        H = self.H0 * np.sqrt((self.Omb + self.Omc) / a ** 3 + (self.Omr + self.Omnu) / a ** 4 + self.OmLambda)
        return H

    def Hmass(self, a):
        # Returns the mass in the Hubble radius
        Hmass = (3. * self.H(a) ** 2 / (8. * math.pi * self.G)) / (self.H(a) / self.c) ** 3;
        return Hmass

    def kk(self, mPBH):
        # Returns the scale (in [Mpc^-1] )  for a given PBH mass
        mH = mPBH / self.ratio_mPBH_over_mH
        kk = self.kmsun * np.sqrt(1. / mH)  # S. Clesse: to check
        return kk

    def kofa(self, a):
        # Returns the scale (in [Mpc^-1] )  for a given scale factor
        kofa = a * self.H(a) / self.c * self.mpc
        return kofa

    def Pk(self, kk):
        # returns the primordial power spectrum of curvature fluctuations at scale kk

        if self.Pk_model == "powerlaw":
            Pk = self.As * (kk / self.kstar) ** (self.ns - 1.) + self.AsPBH * (kk / self.kp) ** (
                        self.nsPBH - 1) * np.heaviside(kk - self.ktrans, 0.5)

        elif self.Pk_model == "lognormal":
            Pk = self.AsPBH * np.exp(- np.log(kk / self.kp) ** 2 / (2 * self.sigma ** 2))

        elif self.Pk_model == "gaussian":
            Pk = self.AsPBH * np.exp(-(kk - self.kp) ** 2 / (2 * (self.sigma * self.kp) ** 2))

        elif self.Pk_model == "broken_powerlaw":
            if (kk < self.kp):
                Pk = self.AsPBH * (kk / self.kp) ** (self.nslow)
            else:
                Pk = self.AsPBH * (kk / self.kp) ** (self.nshigh)

        elif (self.Pk_model == "reheating"):
            Pk = 0.
        else:
            print("Error:  Pk_model does not corresponds to one of the implemented models")
            raise

        return Pk

    # ==================
    # List of fnctions for the computations of delta_critical based on the method Musco20
    def TransferFunction(self, k, t):
        sq = 1. / np.sqrt(3)
        arg = k * t * sq
        return 3 * (np.sin(arg) - arg * np.cos(arg)) / arg ** 3

    def PowerSpectrum(self, k, t):
        return self.Pk(k * self.kp) * self.Pkscalingfactor * self.TransferFunction(k,
                                                                                   t)  # modified by S. Clesse (mutiblied by self.kp)

    def ShapeRHS(self, t, rm=1, print_errors=False):
        cos = lambda k: (k ** 4 * np.cos(k * rm) * self.PowerSpectrum(k, t))
        sin = lambda k: (k ** 4 * np.sin(k * rm) * self.PowerSpectrum(k, t))
        cosint = integrate.quad(cos, 0, np.inf)
        sinint = integrate.quad(sin, 0, np.inf)

        coserr = cosint[1]
        sinerr = sinint[1]

        if print_errors:
            print("errs = ", coserr, sinerr)

        result = -0.5 * (1 + rm * cosint[0] / sinint[0])
        return result

    def F_alpha(self, a):
        arg = 5 / (2 * a)
        diff = (special.gamma(arg) - special.gammainc(arg, 1 / a))
        return np.sqrt(1 - 2 / 5 * np.exp(-1 / a) * a ** (1 - arg) / diff)

    def get_rm(self, t, guess=1, method='root'):

        def func(rm):
            integrand = lambda k: k ** 2 * (
                        (k ** 2 * rm ** 2 - 1) * np.sin(k * rm) / (k * rm) + np.cos(k * rm)) * self.PowerSpectrum(k, t)
            integ = integrate.quad(integrand, 0, np.inf)
            return integ[0]

        if method == 'root':
            sol = opt.root(func, x0=guess)
            root = sol.x
            success = sol.success

            if success:
                return float(root)
            else:
                raise Exception("failed to converge in get_rm iteration")

    def ShapeValue(self, t, rm=1, guess=0.1, method='root'):

        def func(a):
            return self.F_alpha(a) * (1 + self.F_alpha(a)) * a - self.ShapeRHS(t, rm=rm)

        if method == 'root':
            sol = opt.root(func, x0=guess)
            root = sol.x
            success = sol.success

            if success:
                return float(root)
            else:
                raise Exception("failed to converge in ShapeValue iteration")

    def dcrit(self, a):

        if (a >= 0.1 and a <= 3):
            return a ** 0.125 - 0.05
        if (a > 3 and a <= 8):
            return a ** 0.06 + 0.025
        if (a > 8 and a <= 30):
            return 1.15
        # print("  !!! the value of alpha is out of the allowed window (0.1, 30), alpha = {}".format(a))
        raise Exception("  !!! the value of alpha is out of the allowed window (0.1, 30),\n alpha = {}".format(a))

    # ====================

    def read_thermal_file(self):
        # returns a vector of zeta_cr and gthe corresponding vector of m_PBHs

        Npointsinfile = 1501 + 11
        log_m_in_file = np.zeros(Npointsinfile)
        zetacr_in_file = np.zeros(Npointsinfile)
        # Read file of zetacr_thermal as a function of
        fid_file_path = os.path.join(self.data_directory, self.zetacr_thermal_file)
        print("I use the following file for thermal history: ", fid_file_path)
        if os.path.exists(fid_file_path):
            fid_values_exist = True
            with open(fid_file_path, 'r') as fid_file:
                line = fid_file.readline()
                while line.find('#') != -1:
                    line = fid_file.readline()
                while (line.find('\n') != -1 and len(line) == 1):
                    line = fid_file.readline()
                for index_mass in range(Npointsinfile):
                    logmPBH = np.log10(float(line.split()[0]))
                    log_m_in_file[index_mass] = logmPBH  # np.log10(double(line.split()[0])
                    zetacr_in_file[index_mass] = float(line.split()[1])
                    line = fid_file.readline()

        # print log_m_in_file

        return log_m_in_file, zetacr_in_file

    def thermal_factor(self, mPBH):
        # Returns the factor from thermal history by which one has to multiply the zeta_cr obtained for radiation
        logmPBH = np.log10(mPBH)
        zetacr_interp = interp1d(self.log_m_in_file, self.zetacr_in_file, kind='linear')  # , kind='cubic')
        thermal_factor = zetacr_interp(logmPBH) / self.zetacr_thermal_rad
        return thermal_factor

    def logbeta(self, mPBH):
        # returns the density fraction of PBH at formation \beta(m_PBH)
        mH = mPBH / self.ratio_mPBH_over_mH
        kk = self.kmsun / mH ** (0.5)  # S. Clesse: to be checked
        limit_for_using_erfc = 20.
        Pofk = self.Pk(kk) * self.Pkscalingfactor

        if (self.use_thermal_history == True):
            zetacr = self.zetacr_rad * self.thermal_factor(mPBH)
        else:
            zetacr = self.zetacr_rad

        if (self.Gaussian):
            argerfc = zetacr / (2. * Pofk) ** (1. / 2.)
            if (argerfc < limit_for_using_erfc):
                logbeta = np.log10(erfc(argerfc))
            else:
                logbeta = -np.log10(argerfc * np.sqrt(np.pi)) - argerfc ** 2 / np.log(
                    10.)  # S. Clesse Check the logs

        # elif(self.PBHform_model=="standard" and (not(self.Gaussian))):
        return logbeta

    def logfofmPBH(self, mPBH, logbetaform):
        # returns log_10 of the dark matter density fraction of PBH today f(m_PBH)
        mHeq = 2.8e17;
        # logfofmPBH = np.log10((mHeq/(mPBH/self.ratio_mPBH_over_mH))**(1./2.) * 10.**logbetaform * 2./(self.Omc/(self.Omc+self.Omb)))
        logfofmPBH = logbetaform + np.log10(
            (mHeq / (mPBH / self.ratio_mPBH_over_mH)) ** (1. / 2.) * 2. / (self.Omc / (self.Omc + self.Omb)))
        return logfofmPBH

    def foflogmPBH(self, logmPBH):
        # returns the dark matter density fraction of PBH today f(m_PBH)
        logfofmPBH = self.logfofmPBH_interp(logmPBH)
        foflogmPBH = 10. ** logfofmPBH
        return foflogmPBH

    def fPBHtot(self):
        # Computes the integral of the DM density fraction of PBHs today
        fPBHtot = integrate.quad(self.foflogmPBH, self.logmmin, self.logmmax, epsrel=0.001)
        return fPBHtot

    def runPBHform(self):
        # Runs PBH formation
        for ii in range(self.Nmass):
            # print ii
            self.ktable[ii] = self.kk(self.mPBHtable[ii])
            self.Pktable[ii] = self.Pk(self.ktable[ii]) * self.Pkscalingfactor

            if (self.PBHform_model == "Musco20"):
                # print "You chose to comptute zeta_critical with the method Musco20"
                eta = 0.1  # /self.kp  # Modified by S. Clesse:  was initially 0.1
                rm = self.get_rm(eta)
                alpha = self.ShapeValue(eta, rm=rm)
                # print("rm = {}, alpha = {}". format(rm, alpha))
                self.deltacr_rad = self.dcrit(alpha)
                # print("delta_crit = ", self.deltacr_rad)
                self.zetacr_rad = self.deltacr_rad  # * 2.25
                # print("zeta_crit = ", self.zetacr_rad)

            if (self.use_thermal_history == True):
                self.zetacrtable[ii] = self.zetacr_rad * self.thermal_factor(self.mPBHtable[ii])
            else:
                self.zetacrtable[ii] = self.zetacr_rad
            self.logbetatable[ii] = self.logbeta(self.mPBHtable[ii])
            self.logfofmPBHtable[ii] = self.logfofmPBH(self.mPBHtable[ii], self.logbetatable[ii])
            # compute the integrated f_PBH
        self.logfofmPBH_interp = interp1d(self.logmPBHtable, self.logfofmPBHtable)
        logfPBH = np.log10(self.fPBHtot()[0])
        return logfPBH

    def function_to_find_root(self, scaling):
        # function to find the required scaling of the primoridal power spectrum to get a value of f_PBH determined by the user
        self.Pkscalingfactor = 10. ** scaling
        function_to_find_root = self.runPBHform() - np.log10(self.forcedfPBH)
        return function_to_find_root

    # Functions to compute the SGWB from second order perturbations
    def IC2(self, d, s):
        return -36 * math.pi * (s ** 2 + d ** 2 - 2) ** 2 / (s ** 2 - d ** 2) ** 3 * np.heaviside(s - 1, 1)

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

    def rates_primordial(self):
        # Computes the merging rates of primordial binaries
        norm = 1.6e6
        rates = np.zeros((self.Nmass, self.Nmass))
        for ii in range(self.Nmass):
            m1 = self.mPBHtable[ii]
            for jj in range(ii):
                m2 = self.mPBHtable[jj]
                rates[ii, jj] = norm * self.fsup * 10. ** self.logfofmPBHtable[ii] * 10. ** self.logfofmPBHtable[jj] * (
                            m1 + m2) ** (-32. / 37.) * (m1 * m2 / (m1 + m2) ** 2) ** (-34. / 37.)
        return rates

    def rates_clusters(self):
        # Computes the merging rates for tidal capture in PBH clusters
        norm = self.Rclust
        rates = np.zeros((self.Nmass, self.Nmass))
        for ii in range(self.Nmass):
            m1 = self.mPBHtable[ii]
            for jj in range(ii):
                m2 = self.mPBHtable[jj]
                rates[ii, jj] = norm * 10. ** self.logfofmPBHtable[ii] * 10. ** self.logfofmPBHtable[jj] * (
                            m1 + m2) ** (10. / 7.) / (m1 * m2) ** (5. / 7.)
        return rates

    # main function
    def runPBHmodel(self):

        print("Hello, my name is PrimBholes. ")
        print("I am a tool to calculate the gravitational-wave predictions of primordial black holes")
        print("====")
        # Build mode, mass and frequency vectors
        self.logmPBHtable = np.linspace(self.logmmin, self.logmmax, self.Nmass)
        self.logmHtable = self.logmPBHtable - np.log10(self.ratio_mPBH_over_mH)
        self.mPBHtable = 10. ** self.logmPBHtable
        self.mHtable = 10. ** self.logmHtable

        # Read file for thermal history
        self.log_m_in_file, self.zetacr_in_file = self.read_thermal_file()
        # print self.log_m_in_file
        # print self.zetacr_in_file

        # Compute the power spectrum P(k), beta(mPBH) and f(mPBH)
        print("Step 1:  Computation of the power spectrum P(k), the PBH density at formation (beta) and today (f_PBH)")
        print("====")
        self.ktable = np.zeros(self.Nmass)
        self.Pktable = np.zeros(self.Nmass)
        self.logbetatable = np.zeros(self.Nmass)
        self.logfofmPBHtable = np.zeros(self.Nmass)
        self.zetacrtable = np.zeros(self.Nmass)

        if (self.PBHform_model == "standard"):
            print(
                "You chosed to comptute the PBH formation with the fastest standard method (fixed zeta_cr in radiation)")

        if (self.PBHform_model == "Musco20"):
            print("You chosed to comptute zeta_critical with the method Musco20.  It can take several minutes...")

        self.logfPBH = self.runPBHform()
        self.fPBH = 10. ** self.logfPBH

        if (self.PBHform_model == "Musco20"):
            print("delta_crit (radiation) = ", self.deltacr_rad)
            print("zeta_crit (radiation) = ", self.zetacr_rad)

        print("I get a total abundance of PBHs:  f_PBH=", self.fPBH)
        print("with delta_crit (radiation) = ", self.zetacr_rad)  # Wrong no in self
        print("====")


        # Rescale the power spectrum in order to get a fixed value of f_PBH
        if (self.Pkrescaling == True):
            print("Step 1b: Rescaling of the power spectrum to get f_PBH =", self.forcedfPBH)
            sol = opt.bisect(self.function_to_find_root, -1., 1., rtol=1.e-5, maxiter=100)
            self.Pkscalingfactor = 10. ** sol
            self.logfPBH = self.runPBHform()
            self.fPBH = 10. ** self.logfPBH
            print("After rescaling, I get a total abundance of PBHs: fPBH=", self.fPBH)
            print("Rescaling factor=", 10. ** sol)
            print("zeta_crit (radiation) = ", self.zetacr_rad)
            print("====")

        # print self.logfofmPBHtable
        self.ratiodeltacrtable = self.zetacrtable / self.zetacr_rad
        # print self.ratiodeltacrtable

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
        self.freq_2ndOmGW = ks * kvals / 2. / math.pi
        # print "Second coucou 2nd order GW"
        self.OmGW_2ndOmGW = norm * kres

        # ztable =
        ztable = np.linspace(self.zmin, self.zmax, self.Nz)

        # Compute the merging rates
        print("Step 3:  Computation of the PBH merging rates")
        print("====")
        self.rate_prim = np.zeros((self.Nmass, self.Nmass))
        self.rate_clust = np.zeros((self.Nmass, self.Nmass))
        if self.merging_want_primordial == True:
            self.rate_prim = self.rates_primordial()
            print("Merging rates of primordial binaries have been calculated")
            print("====")

        if self.merging_want_clusters == True:
            self.rate_clust = self.rates_clusters()
            print("Merging rates of binaries formed by capture in clusters have been calculated")
            print("====")

        print("End of code, at the moment...  Thank you for having used PrimBholes")

        return self.mPBHtable, self.mHtable, self.ktable, self.Pktable, self.logbetatable, self.logfofmPBHtable, self.fPBH, self.freq_2ndOmGW, self.OmGW_2ndOmGW, self.rate_prim, self.rate_clust



######################################3

test = PrimBHoles()
test.logmmin = -14.
mPBHtable, mHtable, ktable, Pktable, betatable, fofmPBHtable, fPBH, freq_2ndOmGW, OmGW_2ndOmGW, rate_prim, rate_clust = test.runPBHmodel()
#print fPBH
#print freq_2ndOmGW
#print OmGW_2ndOmGW
# print(rate_prim)
# print(rate_clust)


######
_savefig = False

if True:
    figPk = plt.figure()
    figPk.patch.set_facecolor('white')
    ax = figPk.add_subplot(111)
    ax.plot(test.ktable,test.Pktable,label='P(k)')
    plt.title("Primoridal power sepctrum")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(1.e-10,10.)
    plt.xlabel('Wavenumber $k$ [Mpc$^{-1}$]')
    plt.ylabel(r'$\mathcal{P}_{\zeta} (k)$')
    plt.legend(loc='upper left')
    plt.grid(True)
    #ax.patch.set_facecolor('#eeeeee')
    #plt.xlim(test.kp*1.e-6,test.kp*1.e6)
    plt.show()

if True:
    figzetacr = plt.figure()
    figzetacr.patch.set_facecolor('white')
    ax = figzetacr.add_subplot(111)
    ax.plot(test.mHtable,test.ratiodeltacrtable,color='orange')
    #ax.plot(test.mHtable,test.ratiodeltacrtable,label=r'$\zeta_{c}$',color='orange')
    plt.title("Denstiy threshold")
    ax.set_xscale('log')
    plt.ylim(0.85,1.05)
    plt.xlabel(r'$m_{H} [M_\odot]$')
    plt.ylabel(r'$\delta_c / \delta_c^{rad}$')
    #plt.legend(loc='upper left')
    plt.grid(True)
    #ax.patch.set_facecolor('#eeeeee')
    #plt.xlim(test.kp*1.e-6,test.kp*1.e6)
    if _savefig :figzetacr.savefig(figdir + '/deltacr.png',facecolor=figzetacr.get_facecolor(), edgecolor='none',dpi=600)
    plt.show()

if True:
    figbeta = plt.figure()
    figbeta.patch.set_facecolor('white')
    ax = figbeta.add_subplot(111)
    ax.plot(test.mPBHtable,10.**test.logbetatable,label=r'$\beta(m_{PBH})$',color='red')
    plt.title("PBH denstiy distribution at formation")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(1.e-15,1.e-8)
    plt.xlabel(r'$m_{PBH} [M_\odot]$')
    plt.ylabel(r'$\beta(m_{PBH})$')
    plt.legend(loc='upper left')
    plt.grid(True)
    #ax.patch.set_facecolor('#eeeeee')
    #plt.xlim(test.kp*1.e-6,test.kp*1.e6)
    plt.show()

if True:
    figfPBH = plt.figure()
    figfPBH.patch.set_facecolor('white')
    ax = figfPBH.add_subplot(111)
    ax.plot(test.mPBHtable,10.**test.logfofmPBHtable,label=r'$f(m_{PBH})$',color='black')
    plt.title("PBH density distribution today")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(1.e-4,1.e1)
    plt.xlabel(r'$m_{PBH} [M_\odot]$')
    plt.ylabel(r'$f_{PBH}$')
    plt.legend(loc='upper left')
    plt.grid(True)
    #ax.patch.set_facecolor('#eeeeee')
    #plt.xlim(test.kp*1.e-6,test.kp*1.e6)
    plt.show()

if True:
    fig2ndOmGW = plt.figure()
    fig2ndOmGW.patch.set_facecolor('white')
    ax = fig2ndOmGW.add_subplot(111)
    ax.plot(test.freq_2ndOmGW,test.OmGW_2ndOmGW,label=r'$\Omega_{GW}^{\mathrm{2nd Order}}$',color='green')
    plt.title("SGWB from 2nd order perturbations")
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.ylim(1.e-4,1.e1)
    plt.xlabel(r'$f [Hz]$')
    plt.ylabel(r'$\Omega_{GW}$')
    plt.legend(loc='upper left')
    plt.grid(True)
    #ax.patch.set_facecolor('#eeeeee')
    #plt.xlim(test.kp*1.e-6,test.kp*1.e6)
    plt.show()

if True:
    figRprim = plt.figure()
    figRprim.patch.set_facecolor('white')
    ax = figRprim.add_subplot(111)
    zscale = ma.masked_where(test.rate_prim <= 0, test.rate_prim)
    cs=ax.contourf(test.logmPBHtable,test.logmPBHtable, np.transpose(zscale), locator=ticker.LogLocator()) #, cmap=cm.PuBu_r)
    plt.title("Merging rates for primordial binaries")
    cbar = figRprim.colorbar(cs)
    cbar.set_label(r'$yr^{-1}Gpc^{-3}$', rotation=90)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #plt.ylim(1.e-4,1.e1)
    plt.xlabel(r'$\log \, m_1 /M_\odot $')
    plt.ylabel(r'$\log \, m_2 /M_\odot $')
    plt.grid(True)
    if _savefig :figRprim.savefig(figdir + '/Rprim.png',facecolor=figRprim.get_facecolor(), edgecolor='none',dpi=600)
    plt.show()

if True:
    figRclust = plt.figure()
    figRclust.patch.set_facecolor('white')
    ax = figRclust.add_subplot(111)
    zscale = ma.masked_where(test.rate_clust <= 0, test.rate_clust)
    cs=ax.contourf(test.logmPBHtable,test.logmPBHtable, np.transpose(zscale), locator=ticker.LogLocator())
    #cs=ax.pcolormesh(test.logmPBHtable,test.logmPBHtable, np.transpose(zscale), locator=ticker.LogLocator())
    plt.title("Merging rates for tidal capture in clusters")
    cbar = figRclust.colorbar(cs)
    cbar.set_label(r'$yr^{-1}Gpc^{-3}$', rotation=90)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #plt.ylim(1.e-4,1.e1)
    plt.xlabel(r'$\log \, m_1 /M_\odot $')
    plt.ylabel(r'$\log \, m_2 /M_\odot $')
    plt.grid(True)
    if _savefig : figRclust.savefig(figdir + '/Rclust.png',facecolor=figRclust.get_facecolor(), edgecolor='none',dpi=600)
    plt.show()


###############



