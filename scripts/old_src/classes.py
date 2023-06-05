import numpy as np
import os
import scipy.constants as const
import scipy.special as special
from scipy.special import erfc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.integrate import dblquad
import scipy.optimize as opt

from user_params import cosmo_params, physics_units, PBHForm, Pk_models, verbose, MergingRates_models

class ClassCosmology:

    def __init__(self, cp=cosmo_params):
        self.ns = cp.ns
        self.As = cp.As
        self.Omb = cp.Omb
        self.Omc = cp.Omc
        self.h = cp.h
        self.Nur = cp.Nur
        self.TCMB = cp.TCMB
        self.kstar = cp.kstar
        self.H0 = cp.H0
        self.rhocr = cp.rhocr
        self.ar = cp.ar
        self.Omr = cp.Omr
        self.Omnu = cp.Omnu
        self.OmLambda = cp.OmLambda

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
        pu = physics_units
        # Returns the mass in the Hubble radius
        Hmass = (3. * self.H(a) ** 2 / (8. * np.pi * pu.G)) / (self.H(a) / pu.c) ** 3
        return Hmass


class ClassPkSpectrum:

    def __init__(self, pkmodel='default', cm=cosmo_params, pu=physics_units,
                       user_k=None, user_Pk=None):
        name = Pk_models[pkmodel].Pk_model
        self.Pk_model = name            # model name
        self.pkm = Pk_models[name]      # model params
        self.kp = Pk_models.kp          
        self.cm = cm                    # cosmo params
        self.pu = pu                    # physics units

        self.user_k = user_k
        self.user_Pk = user_Pk
        self.interp_func = None

        self.myPkfunction = self.set_Pk(name) 

        if verbose >2: print( 'PkS Model (class) set to ', name)

    def PkUserImport(self, kk):
        Pk = self.interp_func(kk)
        return Pk

    def PkPowerlaw(self, kk):
        pkm = self.pkm
        cm = self.cm
        Pk = cm.As * (kk / cm.kstar) ** (cm.ns - 1.) + pkm.AsPBH * (kk / pkm.kp) ** (
                pkm.nsPBH - 1) * np.heaviside(kk - pkm.ktrans, 0.5)
        return Pk

    def PkLogNormal(self, kk):
        pkm = self.pkm
        Pk = pkm.AsPBH * np.exp(- np.log(kk / pkm.kp) ** 2 / (2 * pkm.sigma ** 2))
        return Pk

    def PkGaussian(self, kk):
        pkm = self.pkm
        Pk = pkm.AsPBH * np.exp(- (kk - pkm.kp)** 2 / (2 * (pkm.sigma * pkm.kp) ** 2))
#         Pk = pkm.AsPBH * np.exp(-(kk - pkm.kp)** 2 / (2 * (pkm.sigma) ** 2))
        return Pk

    def PkBrokenPowerlaw(self, kk):

        if isinstance(kk, float):
            kk = np.array([kk])

        pkm = self.pkm
        # cm = self.cm
        Pk = np.zeros_like(kk)
        mask = (kk < pkm.kc)
        Pk[mask] = pkm.AsPBH_low * (kk[mask] / pkm.kp_low) ** pkm.ns_low
        Pk[~mask] = pkm.AsPBH_high * (kk[~mask] / pkm.kp_high) ** pkm.ns_high
        return Pk

    def PkAxionGauge(self, kk):
        pkm = self.pkm
        cm = self.cm
        As = pkm.As_vac   #cm.As
        PkVac = As * (kk / cm.kstar) ** (cm.ns - 1.)
        PkSource = pkm.AsPBH * np.exp(- np.log(kk / pkm.kp) ** 2 / (2 * pkm.sigma ** 2))
        # TODO:  pkm.AsPBH and pkm.sigma should be k-dependent.
        Pk = PkVac + PkSource
        return Pk

    def PkPreheating(self, kk):

        if isinstance(kk, float):  # necessary when using mask
            kk = np.array([kk])

        pkm = self.pkm
        pu = self.pu
        Pk = np.zeros_like(kk)
        mask = (kk < pkm.kend)
        P0 = pkm.Hstar**2 / (8 * np.pi**2 * pu.m_planck**2  * pkm.e1)
        Pk[~mask] = P0
        Pk[mask] = pkm.Hstar**2 / (8 * np.pi**2 * pu.m_planck**2  * pkm.e1) * \
                  (1 + (kk[mask]/pkm.kend)**2) * (1 - 2*(pkm.C+1)*pkm.e1 - pkm.C * pkm.e2)
        # TODO: pkm.Hstar, pkm.e1 and pkm.e2  should be k-dependent.

        # print(Pk)
        return Pk


    def PkMultifield(self, kk):

        if isinstance(kk, float):  # necessary when using mask
            kk = np.array([kk])

        pkm = self.pkm
        Pk = np.zeros_like(kk)
        kappa = kk/pkm.kf
        arg = (2-kappa)*kappa
        mask = (kappa < 1.8)

        Pk[~mask] = pkm.P0
        Pk[mask] = pkm.P0 * np.exp(2*np.sqrt(arg[mask])*pkm.eta*pkm.delta) / (2*arg[mask]) * \
            np.sin(np.exp(-pkm.delta/2)*kappa[mask]*pkm.eta + np.arctan(kappa[mask]/np.sqrt(arg[mask])))
        return Pk


    def set_Pk(self, name, user_k=None, user_Pk=None):
        # returns the primordial power spectrum of curvature fluctuations at scale kk
        self.Pk_model = name
        self.pkm = Pk_models[name]

        if name == "powerlaw":
            return self.PkPowerlaw

        elif name == "lognormal":
            return self.PkLogNormal

        elif name == "gaussian":
            return self.PkGaussian

        elif name == "broken_powerlaw":
            return self.PkBrokenPowerlaw

        elif name == "axion_gauge":
            return self.PkAxionGauge

        elif name == "preheating":
            return self.PkPreheating

        elif name == "multifield":
            return self.PkMultifield
        
        elif name == "user_import":   
            user_k = self.user_k if not user_k else user_k
            user_Pk = self.user_Pk if not user_Pk else user_Pk
            self.interp_func = interp1d(user_k, user_Pk)         
            return self.PkUserImport

        else:
            myError = "Error:  Pk_model does not corresponds to one of the implemented models"
            print(myError)
            raise Exception(myError) 

    def set_Pk_function(self, func):
        self.myPkfunction = func
        return True
    
    def Pk(self, kk):
        # returns the primordial power spectrum of curvature fluctuations at scale kk

        return self.myPkfunction(kk)


class ClassPBHFormationMusco20:

    def __init__(self, pmm=PBHForm.models.Musco20,                       # TODO: Only used for eta!  
                       Pk_model='default',                               # TODO: Import just P(k) function? 
                       cp=cosmo_params):
        self.eta = pmm.eta
        self.Pk_model = Pk_models[Pk_model].Pk_model
        self.kp = Pk_models.kp
        self.cp = cp
        self.Pkscalingfactor = PBHForm.Pkscalingfactor

        if verbose >2: print( 'FormPBHMusco (class) set  Pk to ', Pk_model)

    # List of fnctions for the computations of delta_critical based on the method Musco20
    def TransferFunction(self, k, t):
        sq = 1. / np.sqrt(3)
        arg = k * t * sq
        return 3 * (np.sin(arg) - arg * np.cos(arg)) / arg ** 3

    def PowerSpectrum(self, k, t):
        PkS = ClassPkSpectrum(pkmodel=self.Pk_model, cm=self.cp)
        P = PkS.Pk(k * self.kp) * self.Pkscalingfactor
        T =  self.TransferFunction(k, t)
        return 2.*np.pi**2 /(k**3) * P * T**2

    def ShapeRHS(self, t, rm=1.0, print_errors=False):
        cos = lambda k: (k ** 4 * np.cos(k * rm) * self.PowerSpectrum(k, t))
        sin = lambda k: (k ** 3 * np.sin(k * rm) * self.PowerSpectrum(k, t))
        cosint = integrate.quad(cos, 0, np.inf, limit=100000, limlst=10000)
        sinint = integrate.quad(sin, 0, np.inf, limit=100000, limlst=10000)
        coserr = cosint[1]
        sinerr = sinint[1]

        if print_errors:
            print("errs = ", coserr, sinerr)

        result = -0.25 * (1 + rm * cosint[0] / sinint[0])
        return result

    def F_alpha(self, a):
        arg = 5 / (2 * a)

        # gammainc  ==  inc lower gammar   Normalized  (1/gamma(arg))
        # gammaincc ==  inc upper gammar  Normalized (1/gamma(arg))
        diff = (special.gamma(arg) * special.gammainc(arg, 1 / a))

        if not diff==diff:
            print('diff = ', diff,
                  " \n a=5/(2*alpha), x = 1/alpha , G(a), G_inc(a, x) = ",
                  arg, 1/a, special.gamma(arg),  special.gammainc(arg, 1 / a),  "\n\n" )
            raise Exception('  !!  nan at calc Gammas in F(alpha)')

        disc = 1 - 2 / 5 * np.exp(-1 / a) * a ** (1 - arg) / diff

        if disc < 0 or not disc==disc:
            print('disc = ', disc, "\n\n" )
            raise Exception('  !!  negative  sqrt  or nan in F(alpha)')

        return np.sqrt(disc)

    def get_rm(self, t, guess=1.0, method='root'):

        def func(rm):
            integrand = lambda k: k ** 2 * (
                        (k ** 2 * rm ** 2 - 1) * np.sin(k * rm) / (k * rm) + np.cos(k * rm)) \
                                  * self.PowerSpectrum(k, t)
            integ = integrate.quad(integrand, 0, np.inf, limit=100000, limlst=10000)
            return integ[0]

        if method == 'root':
            sol = opt.root(func, x0=guess)
            root = sol.x
            success = sol.success

        else:
            root, success = opt.bisect(func, 0., 1e8, rtol=1.e-5, maxiter=100, full_output = True)
            # success = True

        # print('\n\n\n kp , rm , kp*rm=', root ,  self.kp, root * self.kp )  #TODO :clean

        if success:
                return float(root)
        else:
                raise Exception("failed to converge in get_rm iteration")

    def ShapeValue(self, t, rm=1.0, guess=0.5, method='root'):

        def func(a):
            return self.F_alpha(a) * (1 + self.F_alpha(a)) * a - 2 * self.ShapeRHS(t, rm=rm)

        # method = 'bisect' #TODO

        if method == 'root':
            sol = opt.root(func, x0=guess)
            root = sol.x
            success = sol.success
        else:
            root, success = opt.bisect(func, 0.01, 100., rtol=1.e-5, maxiter=100, full_output = True)

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
        raise Exception("  !!! the value of alpha is out of the allowed window (0.1, 30),\n alpha = {}".format(a))

    def get_deltacr(self):
        # print("getting delta_cr with eta =  ", self.eta,  " kp =", self.kp) #TODO :clean
        eta = self.eta        # /self.kp  # Modified by S. Clesse:  was initially 0.1
        rm = self.get_rm(eta)
        alpha = self.ShapeValue(eta, rm=rm)
        deltacr = self.dcrit(alpha)
        return deltacr



class ClassPBHFormation:

    def __init__(self, pm=PBHForm, cp=cosmo_params, Pk_model='default', PBHform_model='default'):
        if PBHform_model == 'default':
            PBHform_model = pm.PBHform_model
        self.pm = pm
        self.cp = cp
        self.ratio_mPBH_over_mH = pm.ratio_mPBH_over_mH
        self.kmsun = pm.kmsun
        self.Pkscalingfactor = pm.Pkscalingfactor
        self.use_thermal_history = pm.use_thermal_history
        self.Pkrescaling = pm.Pkrescaling
        self.rescaling_is_done = False
        self.forcedfPBH = pm.forcedfPBH
        self.use_thermal_history = pm.use_thermal_history
        self.data_directory = pm.data_directory
        self.zetacr_thermal_file = pm.zetacr_thermal_file
        self.zetacr_thermal_rad = pm.zetacr_thermal_rad
        self.Gaussian = pm.Gaussian
        self.Pk_model = Pk_model
        self.PBHform_model = pm.models[PBHform_model].PBHform_model

        if verbose:
            print('FormPBH (class) set  PBHform_model to ', self.PBHform_model)
            print('FormPBH (class) set  Pk to ', Pk_model)


    def get_thermalfactor(self, mPBH):
        # Read file
        def _read_thermal_file(datadir, zetacr_path): #TODO (put as a separete class?)

            # returns a vector of zeta_cr and gthe corresponding vector of m_PBHs
            Npointsinfile = 1501 + 11
            log_m_in_file = np.zeros(Npointsinfile)
            zetacr_in_file = np.zeros(Npointsinfile)
            # Read file of zetacr_thermal as a function of
            fid_file_path = os.path.join(datadir, zetacr_path)
            if verbose > 2 : print("I use the following file for thermal history: ", fid_file_path)
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

            return log_m_in_file, zetacr_in_file

        log_m_in_file, zetacr_in_file = _read_thermal_file(self.data_directory, self.zetacr_thermal_file)
        # Returns the factor from thermal history by which one has to multiply the zeta_cr obtained for radiation
        zetacr_interp = interp1d(log_m_in_file, zetacr_in_file, kind='linear')  # , kind='cubic')
        logmPBH = np.log10(mPBH)
        thermal_factor = zetacr_interp(logmPBH) / self.zetacr_thermal_rad
        return thermal_factor

    def get_deltacr(self, method='default'):


        if (method == 'default'):
            method = self.PBHform_model

        # Depends method
        if (method=='standard'):
            deltacr_rad =self.pm.models.standard.zetacr_rad
        elif (method == "Musco20"):
            pmm = ClassPBHFormationMusco20(Pk_model=self.Pk_model, cp=self.cp)
            deltacr_rad = pmm.get_deltacr()
        else:
            mess = 'Such method ({m}) is still not implemented yet'.format(m=method)
            print(mess)
            raise Exception(mess)

        return deltacr_rad

    def get_zetacr(self, mPBH, method='default', use_thermal_history='default'):

        if (method == 'default'):
            method = self.PBHform_model
        if (use_thermal_history == 'default'):
            use_thermal_history = self.use_thermal_history

        if self.Pkrescaling == True and self.rescaling_is_done == False:
            self.calc_scaling(mPBH)

        zetacr_rad = self.get_deltacr(method=method)

        # Usage of thermal history
        if (use_thermal_history == True):
            zetacrtable = zetacr_rad * self.get_thermalfactor(mPBH)
        else:
            zetacrtable = zetacr_rad

        return zetacrtable

    def get_logbeta(self, mPBH, use_thermal_history='default'):  #TODO

        if use_thermal_history == 'default':
            use_thermal_history = self.use_thermal_history

        if self.Pkrescaling == True and self.rescaling_is_done == False:
            self.calc_scaling(mPBH)

        PkS = ClassPkSpectrum(pkmodel=self.Pk_model, cm=self.cp)

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
            raise (ValueError, "Non-Gaussian perturbations are not implemented yet")

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
        return ClassPkSpectrum(pkmodel=self.Pk_model, cm=self.cp).Pk(k)


    #############################################################################
    ###########################################################################
    #Non-understood functions  #TODO: order/take out into other class

    ###################################
    # Functions to compute the SGWB from second order perturbations
    def IC2(self, d, s):
        return -36 * np.pi * (s ** 2 + d ** 2 - 2) ** 2 / (s ** 2 - d ** 2) ** 3 * np.heaviside(s - 1, 1)

    def IS2(self, d, s):
        return -36 * (s ** 2 + d ** 2 - 2) / (s ** 2 - d ** 2) ** 2 * (
                (s ** 2 + d ** 2 - 2) / (s ** 2 - d ** 2) * np.log((1 - d ** 2) / np.absolute(s ** 2 - 1)) + 2)

    def IcsEnvXY(self, x, y):
        return (self.IC2(np.absolute(x - y) / (3 ** 0.5), np.absolute(x + y) / (3 ** 0.5)) ** 2 + self.IS2(
            np.absolute(x - y) / (3 ** 0.5), np.absolute(x + y) / (3 ** 0.5)) ** 2) ** 0.5

    def compint(self, kvval, sigmaps):
        PkS = ClassPkSpectrum(pkmodel=self.Pk_model, cm=self.cp)
        pu = physics_units
        value, error = dblquad(lambda x, y:
                               x ** 2 / y ** 2 * (1 - (1 + x ** 2 - y ** 2) ** 2 / (4 * x ** 2)) ** 2
                               * PkS.Pk(kvval * pu.mpc / pu.c) * self.Pkscalingfactor  # PS(kvval*x)
                               * PkS.Pk(kvval * pu.mpc / pu.c) * self.Pkscalingfactor  # PS(kvval*y)
                               * self.IcsEnvXY(x, y) ** 2
                               ,
                               10 ** (- 4 * sigmaps) / kvval, 10 ** (4 * sigmaps) / kvval, lambda x: np.absolute(1 - x),
                               lambda x: 1 + x)
        return value


    #########################################

    def rates_primordial(self, mPBHtable, fofmPBHtable, triangular=True):
        # Computes the merging rates of primordial binaries
        fsup = MergingRates_models.primordial.fsup
        norm = MergingRates_models.primordial.norm
        m1, m2 = np.meshgrid(mPBHtable, mPBHtable)
        f1, f2 = np.meshgrid(fofmPBHtable, fofmPBHtable)
        rates = norm * fsup * f1 * f2 * (m1 + m2)**(-32. / 37.) * (m1 * m2 / (m1 + m2)** 2)**(-34. / 37.)

        if triangular: rates = np.tril(rates)

        return rates


    def rates_clusters(self, mPBHtable, fofmPBHtable, triangular=True):
        # Computes the merging rates for tidal capture in PBH clusters
        Rclust = MergingRates_models.clusters.Rclust
        m1, m2 = np.meshgrid(mPBHtable, mPBHtable)
        f1, f2 = np.meshgrid(fofmPBHtable, fofmPBHtable)

        rates = Rclust * f1 * f2 * (m1 + m2) ** (10. / 7.) / (m1 * m2) ** (5. / 7.)

        if triangular: rates = np.tril(rates)

        return rates



#######################################

if __name__ == "__main__":
    # Munch allows to set-up/extract values as  both dictionary and class ways
    print(physics_units.c, physics_units['c'])
